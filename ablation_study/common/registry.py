from dataclasses import dataclass


@dataclass(frozen=True)
class AblationVariant:
    name: str
    description: str


COT_MACRO_VARIANTS = {
    "full": AblationVariant("full", "history + TD -> preference -> macro"),
    "w_o_td": AblationVariant("w_o_td", "remove time discretization and rebuild preference"),
    "w_o_preference": AblationVariant("w_o_preference", "remove preference when predicting macro"),
    "history_only": AblationVariant("history_only", "raw history only for macro prediction"),
}


COT_FINE_VARIANTS = {
    "full": AblationVariant("full", "history + preference + macro -> fine"),
    "w_o_td": AblationVariant("w_o_td", "remove time discretization and rebuild preference"),
    "w_o_preference": AblationVariant("w_o_preference", "history + macro -> fine"),
    "w_o_macro": AblationVariant("w_o_macro", "history + preference -> fine"),
    "history_only": AblationVariant("history_only", "raw history only for fine prediction"),
}


FINAL_LLM_VARIANTS = {
    "full": AblationVariant("full", "history + preference + macro + fine -> poi"),
    "w_o_fine": AblationVariant("w_o_fine", "remove fine in final POI prediction"),
    "w_o_macro": AblationVariant("w_o_macro", "remove macro through the end-to-end chain"),
    "w_o_preference": AblationVariant("w_o_preference", "remove preference through the end-to-end chain"),
    "w_o_td": AblationVariant("w_o_td", "remove TD from upstream chain and reuse fine w_o_td outputs"),
    "input_masking": AblationVariant("input_masking", "mask one semantic field only at final prompt"),
}
