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

s_neuron = genn_model.create_custom_neuron_class(
    "S",
    param_names=["tau_m", "tau_in", "v_norm"],
    var_name_types=[
        ("dx", "scalar"),
        ("dy", "scalar"),
        ("vt", "scalar"),
        ("dt", "scalar"),
        ("vx", "scalar"),
        ("vy", "scalar"),
        ("V", "scalar"),
    ],
    sim_code="""
    $(dx) += DT * ($(Isyn_x) - $(dx)) / $(tau_in);
    $(dy) += DT * ($(Isyn_y) - $(dy)) / $(tau_in);
    $(dt) = ($(Isyn_t) - $(vt)) / $(tau_in);
    $(vt) += DT * $(dt);

    const scalar sgn_dx = $(dx) > 0.0 ? 1.0 : -1.0;
    const scalar sgn_dy = $(dy) > 0.0 ? 1.0 : -1.0;

    $(vx) = -sgn_dx * $(dt) / (abs($(dx)) + $(v_norm));
    $(vy) = -sgn_dy * $(dt) / (abs($(dy)) + $(v_norm));

    const scalar v_proj = $(xnorm)[$(id)] * $(vx) + $(ynorm)[$(id)] * $(vy);

    $(V) += DT * (v_proj - $(V)) / $(tau_m);
    """,
    extra_global_params=[("xnorm", "scalar*"), ("ynorm", "scalar*")],
    additional_input_vars=[
        ("Isyn_x", "scalar", 0.0),
        ("Isyn_y", "scalar", 0.0),
        ("Isyn_t", "scalar", 0.0),
    ],
    is_auto_refractory_required=False,
)

def reexp(x):
    return f"(1.0 - exp(-max(0.0, {x})))"

out_neuron = genn_model.create_custom_neuron_class(
    "OUT",
    param_names=["g_filt_bias", "g_filt_scale", "output_scale"],
    var_name_types=[("S_left", "scalar"), ("S_right", "scalar"), ("V", "scalar")],
    sim_code=f"""
    $(S_left) = $(Isyn_left);
    $(S_right) = $(Isyn_right);

    const scalar g_filt_left = {reexp('($(S_left) - $(g_filt_bias)) / $(g_filt_scale)')};
    const scalar g_filt_right = {reexp('($(S_right) - $(g_filt_bias)) / $(g_filt_scale)')};

    //$(V) = 0.5 * ($(S_left) + $(S_right)) * $(output_scale);

    $(V) = g_filt_left * g_filt_right * 0.5 * ($(S_left) + $(S_right)) * $(output_scale);
    """,
    additional_input_vars=[("Isyn_left", "scalar", 0.0), ("Isyn_right", "scalar", 0.0)],
)

cont_wu = genn_model.create_custom_weight_update_class(
    "cont_wu",
    param_names=[],
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code="$(addToInSyn, $(g) * $(V_pre));"
)
