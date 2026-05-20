from dataclasses import dataclass


@dataclass
class CotStageFlags:
    use_time_discretization: bool = True
    use_preference: bool = True
    use_macro: bool = True
    history_only: bool = False


def flags_for_macro_variant(variant: str) -> CotStageFlags:
    if variant == "full":
        return CotStageFlags()
    if variant == "w_o_td":
        return CotStageFlags(use_time_discretization=False)
    if variant == "w_o_preference":
        return CotStageFlags(use_preference=False)
    if variant == "history_only":
        return CotStageFlags(use_time_discretization=False, use_preference=False, history_only=True)
    raise ValueError(f"Unsupported macro variant: {variant}")


def flags_for_fine_variant(variant: str) -> CotStageFlags:
    if variant == "full":
        return CotStageFlags()
    if variant == "w_o_td":
        return CotStageFlags(use_time_discretization=False)
    if variant == "w_o_preference":
        return CotStageFlags(use_preference=False)
    if variant == "w_o_macro":
        return CotStageFlags(use_macro=False)
    if variant == "history_only":
        return CotStageFlags(use_time_discretization=False, use_preference=False, use_macro=False, history_only=True)
    raise ValueError(f"Unsupported fine variant: {variant}")
