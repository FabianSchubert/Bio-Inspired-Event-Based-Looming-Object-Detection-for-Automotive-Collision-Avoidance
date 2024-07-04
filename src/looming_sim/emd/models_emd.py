from pygenn import genn_model

p_neuron = genn_model.create_custom_neuron_class(
    "P",
    param_names=[],
    var_name_types=[("V", "scalar")],
    sim_code="$(V) = $(Isyn);",
    threshold_condition_code="""
    $(V) > 0.0
    """,
    reset_code="",
    is_auto_refractory_required=False,
)

g_neuron = genn_model.create_custom_neuron_class(
    "G",
    param_names=["tau_in"],
    var_name_types=[
        ("dx", "scalar"),
        ("dy", "scalar"),
        ("dt", "scalar"),
        ("vt", "scalar"),
        ("dxdx", "scalar"),
        ("dydy", "scalar"),
        ("dxdy", "scalar"),
        ("dxdt", "scalar"),
        ("dydt", "scalar"),
    ],
    sim_code="""
    $(dx) += DT * ($(Isyn_x) - $(dx)) / $(tau_in);
    $(dy) += DT * ($(Isyn_y) - $(dy)) / $(tau_in);
    $(dt) = ($(Isyn_t) - $(vt)) / $(tau_in);
    $(vt) += DT * $(dt);

    $(dxdx) = $(dx) * $(dx);
    $(dydy) = $(dy) * $(dy);
    $(dxdy) = $(dx) * $(dy);
    $(dxdt) = $(dx) * $(dt);
    $(dydt) = $(dy) * $(dt);
    """,
    additional_input_vars=[
        ("Isyn_x", "scalar", 0.0),
        ("Isyn_y", "scalar", 0.0),
        ("Isyn_t", "scalar", 0.0),
    ],
    is_auto_refractory_required=False,
)

u_neuron = genn_model.create_custom_neuron_class(
    "U",
    param_names=["reg", "tau_m"],
    var_name_types=[("ux", "scalar"), ("uy", "scalar"), ("V", "scalar")],
    sim_code="""
    const scalar a00 = $(Isyn_dxdx) + $(reg);
    const scalar a01 = $(Isyn_dxdy);
    const scalar a10 = $(Isyn_dxdy);
    const scalar a11 = $(Isyn_dydy) + $(reg);

    const scalar b0 = -$(Isyn_dxdt);
    const scalar b1 = -$(Isyn_dydt);

    const scalar det = a00 * a11 - a01 * a10;

    $(ux) = (a11 * b0 - a01 * b1) / det;
    $(uy) = (a00 * b1 - a10 * b0) / det;

    const scalar u_proj = $(xnorm)[$(id)] * $(ux) + $(ynorm)[$(id)] * $(uy);
    $(V) += DT * (u_proj - $(V)) / $(tau_m);
    """,
    extra_global_params=[("xnorm", "scalar*"), ("ynorm", "scalar*")],
    additional_input_vars=[
        ("Isyn_dxdx", "scalar", 0.0),
        ("Isyn_dydy", "scalar", 0.0),
        ("Isyn_dxdy", "scalar", 0.0),
        ("Isyn_dxdt", "scalar", 0.0),
        ("Isyn_dydt", "scalar", 0.0),
    ],
    is_auto_refractory_required=False,
)

def reexp(x):
    return f"(1.0 - exp(-max(0.0, {x})))"

out_neuron = genn_model.create_custom_neuron_class(
    "OUT",
    param_names=["g_filt_bias", "g_filt_scale", "output_scale"],
    var_name_types=[("U_left", "scalar"), ("U_right", "scalar"), ("V", "scalar")],
    sim_code=f"""
    $(U_left) = $(Isyn_left);
    $(U_right) = $(Isyn_right);

    const scalar g_filt_left = {reexp('($(U_left) - $(g_filt_bias)) / $(g_filt_scale)')};
    const scalar g_filt_right = {reexp('($(U_right) - $(g_filt_bias)) / $(g_filt_scale)')};

    $(V) = g_filt_left * g_filt_right * 0.5 * ($(U_left) + $(U_right)) * $(output_scale);
    """,
    additional_input_vars=[("Isyn_left", "scalar", 0.0), ("Isyn_right", "scalar", 0.0)],
)

cont_wu = genn_model.create_custom_weight_update_class(
    "cont_wu",
    param_names=[],
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code="$(addToInSyn, $(g) * $(V_pre));"
)

def create_cont_wu(name: str, presyn_var: str):
    return genn_model.create_custom_weight_update_class(
        name,
        param_names=[],
        var_name_types=[("g", "scalar")],
        synapse_dynamics_code=f"$(addToInSyn, $(g) * $({presyn_var}_pre));"
    )
