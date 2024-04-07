from pygenn import genn_model

p_neuron = genn_model.create_custom_neuron_class(
    "P",
    param_names=["tau_m", "tau_i", "V_thresh", "V_reset"],
    var_name_types=[("V", "scalar"), ("VI", "scalar")],
    sim_code="""
    $(V) += DT * ($(VI) - $(V)) / $(tau_m);
    $(VI) += DT * ($(Isyn) - $(VI)) / $(tau_i);
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    $(V) = $(V_reset);  // reset to $(V_reset)
    """,
    is_auto_refractory_required=False,
)

s_neuron = genn_model.create_custom_neuron_class(
    "S",
    param_names=["tau_m", "vel_norm"],
    var_name_types=[
        ("Vx", "scalar"),
        ("Vy", "scalar"),
        ("V", "scalar"),
    ],
    sim_code="""
    $(Vx) = $(Isyn_x) * $(Isyn_t) / ($(vel_norm) + $(Isyn_norm));
    $(Vy) = $(Isyn_x) * $(Isyn_t) / ($(vel_norm) + $(Isyn_norm));

    const scalar v_proj = $(x)[$(id)] * $(Vx) + $(y)[$(id)] * $(Vy);

    const scalar g_est = v_proj;

    $(V) += DT * (g_est - $(V)) / $(tau_m);
    """,
    extra_global_params=[("x", "scalar*"), ("y", "scalar*")],
    additional_input_vars=[
        ("Isyn_x", "scalar", 0.0),
        ("Isyn_y", "scalar", 0.0),
        ("Isyn_t", "scalar", 0.0),
        ("Isyn_norm", "scalar", 0.0),
    ],
    is_auto_refractory_required=False,
)

out_neuron = genn_model.create_custom_neuron_class(
    "OUT",
    param_names=["tau_m", "g_filt_bias", "g_filt_scale"],
    var_name_types=[("S_left", "scalar"), ("S_right", "scalar"), ("V", "scalar")],
    sim_code="""
   $(S_left) = $(Isyn_left);
   $(S_right) = $(Isyn_right);
   //const scalar g_filt_left = 1./(1.+exp(-4.*($(S_left)-$(g_filt_bias))/$(g_filt_scale)));
   //const scalar g_filt_right = 1./(1.+exp(-4.*($(S_right)-$(g_filt_bias))/$(g_filt_scale)));
   
   //const scalar g_filt_left = $(S_left) > 0.0 ? 1.0 : 0.0;
   //const scalar g_filt_right = $(S_right) > 0.0 ? 1.0 : 0.0;

   const scalar g_filt_left = 1.0 - exp(-max(0.0, ($(S_left) - $(g_filt_bias)))/$(g_filt_scale));
   const scalar g_filt_right = 1.0 - exp(-max(0.0, ($(S_right) - $(g_filt_bias)))/$(g_filt_scale));
   
   $(V) += DT * (g_filt_left * g_filt_right * 0.5 * ($(S_left) + $(S_right)) - $(V)) / $(tau_m);
   """,
    additional_input_vars=[("Isyn_left", "scalar", 0.0), ("Isyn_right", "scalar", 0.0)],
)

cont_wu = genn_model.create_custom_weight_update_class(
    "cont_wu",
    param_names=[],
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code="$(addToInSyn, $(g) * $(V_pre));"
)
