"""Streamlit entry point for the HTTP-only deterministic smoke demo."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import streamlit as st
from pydantic import ValidationError

from next_poi.contracts import HistoryEvent, RecommendationRequest, RecommendationResponse
from next_poi.demo.api_client import (
    ApiClient,
    ApiClientError,
    ApiUnavailableError,
    ApiValidationError,
)

API_BASE_URL_ENV = "NEXT_POI_API_BASE_URL"
DEFAULT_API_BASE_URL = "http://api:8000"


def parse_history(value: str) -> tuple[HistoryEvent, ...]:
    """Parse ``POI | category | aware ISO-8601 time`` lines through contracts."""

    events: list[HistoryEvent] = []
    for line_number, raw_line in enumerate(value.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 3:
            raise ValueError(f"第 {line_number} 行应包含 POI、类别和时间三列")
        poi_id, category_name, timestamp = parts
        try:
            events.append(
                HistoryEvent(
                    poi_id=poi_id,
                    category_name=category_name,
                    timestamp=timestamp,
                )
            )
        except ValidationError as exc:
            raise ValueError(f"第 {line_number} 行格式无效") from exc
    if not events:
        raise ValueError("历史轨迹不能为空")
    return tuple(events)


def build_request(
    *,
    dataset: str,
    history_text: str,
    target_time: datetime,
    top_k: int,
) -> RecommendationRequest:
    """Build the canonical target-blind request used by the API client."""

    return RecommendationRequest(
        dataset=dataset,
        history=parse_history(history_text),
        target_time=target_time,
        top_k=top_k,
        profile="smoke",
    )


def render_service_status(client: ApiClient) -> None:
    st.subheader("服务状态")
    try:
        client.health()
        ready = client.ready()
        version = client.version()
    except ApiUnavailableError as exc:
        st.warning(f"API 尚未就绪：{exc.message}")
        return
    except ApiClientError as exc:
        st.error(f"无法读取服务状态：{exc.message}")
        return
    st.success("CPU smoke bundle 已校验并就绪")
    st.caption(
        f"profile={ready['profile']} · release={version.release} · "
        f"data={version.data} · model={version.model}"
    )


def render_recommendations(response: RecommendationResponse) -> None:
    st.subheader("推荐结果")
    if not response.recommendations:
        st.info("当前请求没有返回推荐项。")
        return
    st.dataframe(
        [
            {
                "rank": item.rank,
                "poi_id": item.poi_id,
                "category": item.category,
                "score": item.score,
                "sources": ", ".join(item.candidate_sources),
            }
            for item in response.recommendations
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        f"request_id={response.request_id} · total={response.latency.total_ms:.3f} ms · "
        f"release={response.versions.release} · model={response.versions.model}"
    )


def main() -> None:
    st.set_page_config(page_title="Next-POI CPU Smoke Demo", layout="wide")
    st.title("Next-POI CPU Smoke Demo")
    st.caption("本页面只调用 FastAPI；结果不代表 full-GPU、线上流量或论文指标。")

    base_url = os.environ.get(API_BASE_URL_ENV, DEFAULT_API_BASE_URL)
    try:
        client = ApiClient(base_url)
    except ValueError:
        st.error(f"{API_BASE_URL_ENV} 必须是有效的 HTTP(S) 地址。")
        return

    try:
        render_service_status(client)
        dataset = st.selectbox("数据集", ("synthetic", "nyc", "tky", "ca"))
        history_text = st.text_area(
            "历史轨迹（每行：POI | 类别 | 含时区 ISO-8601 时间）",
            value="unknown-poi | unknown-category | 2026-01-01T00:00:00+00:00",
            height=130,
        )
        target_date = st.date_input("目标日期", value=datetime(2026, 1, 1).date())
        target_clock = st.time_input("目标时间（UTC）", value=datetime.min.time())
        top_k = st.slider("Top-K", min_value=1, max_value=100, value=5)

        running = bool(st.session_state.get("recommendation_running", False))
        submitted = st.button("获取推荐", type="primary", disabled=running)
        if submitted:
            st.session_state["recommendation_running"] = True
            try:
                request = build_request(
                    dataset=dataset,
                    history_text=history_text,
                    target_time=datetime.combine(
                        target_date, target_clock, tzinfo=timezone.utc
                    ),
                    top_k=top_k,
                )
                with st.spinner("正在请求 smoke API…"):
                    response = client.recommend(request)
                render_recommendations(response)
            except ValueError as exc:
                st.error(f"输入无效：{exc}")
            except ValidationError:
                st.error("请求字段校验失败，请检查目标时间、历史长度和 Top-K。")
            except ApiValidationError as exc:
                st.error(f"API 拒绝了请求：{exc.message}")
            except ApiUnavailableError as exc:
                st.warning(f"API 不可用：{exc.message}")
            except ApiClientError as exc:
                st.error(f"API 调用失败：{exc.message}")
            finally:
                st.session_state["recommendation_running"] = False
    finally:
        client.close()


if __name__ == "__main__":
    main()
