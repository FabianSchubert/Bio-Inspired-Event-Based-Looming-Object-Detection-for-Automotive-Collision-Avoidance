from pygenn import genn_model

# curent source model for current input based on DVS events passed as a spike_array
spike_array_current_source = genn_model.create_custom_current_source_class(
    "",
    param_names=["unit_amplitude"],
    var_name_types=[("startSpike", "int"), ("endSpike", "int")],
    extra_global_params=[("spikeTimes", "scalar*"), ("spikePolarity", "int8_t*")],
    injection_code="""
    if ($(startSpike) != $(endSpike) && $(t) >= $(spikeTimes)[$(startSpike)]) {
        $(injectCurrent, $(unit_amplitude)*$(spikePolarity)[$(startSpike)]);
        $(startSpike)++;
    }
    if($(startSpike) > $(endSpike)){
        printf("Warning, startspike > endspike (should never happen)");
    }
    """,
)

bitmask_array_current_source = genn_model.create_custom_current_source_class(
    "",
    param_names=["unit_amplitude"],
    var_name_types=[("nt", "int"), ("pop_size", "int")],
    extra_global_params=[
        ("spikeBitmask", "uint32_t*"),
        ("polarityBitmask", "uint32_t*"),
    ],
    injection_code="""

    const unsigned int wordsPerTimestep = (($(pop_size) + 31) / 32);
    const uint32_t mySpikeWord = $(spikeBitmask)[(wordsPerTimestep * (int)round($(t) / DT)) + $(id) / 32];
    const uint32_t myPolarityWord = $(polarityBitmask)[(wordsPerTimestep * (int)round($(t) / DT)) + $(id) / 32];

    if(mySpikeWord & (1 << ($(id) % 32))){
        
        const int pol = (myPolarityWord & (1 << ($(id) % 32))) ? 1 : -1;
        $(injectCurrent,
            $(unit_amplitude)*(pol * 2.0 - 1.0)
        );
    }
    """,
)

# LIF neuron model for S neurons with soft reset as in Banchard et al. 2000
lif_neuron = genn_model.create_custom_neuron_class(
    "LIF",
    param_names=["tau_m", "V_thresh", "V_reset"],
    var_name_types=[("V", "scalar"), ("VI", "scalar")],
    sim_code="""
    $(VI) = $(Isyn);
    $(V) += ($(Isyn)-$(V))/$(tau_m)*DT;  // linear Euler
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    $(V)-= $(V_reset);  // soft reset by $(V_reset)
    """,
    is_auto_refractory_required=False,
)

s_neuron = genn_model.create_custom_neuron_class(
    "S",
    param_names=["tau_m", "tau_filt", "b_reg"],
    var_name_types=[
        ("Vx", "scalar"),
        ("Vy", "scalar"),
        ("Vt", "scalar"),
        ("It_prev", "scalar"),
        ("V", "scalar"),
    ],
    sim_code="""
    $(Vx) += DT * ($(Isyn_x) - $(Vx)) / $(tau_filt);
    $(Vy) += DT * ($(Isyn_y) - $(Vy)) / $(tau_filt);
    $(Vt) += DT * (($(Isyn_t) - $(It_prev))/DT - $(Vt)) / $(tau_filt);

    $(It_prev) = $(Isyn_t);

    const scalar v_proj = $(xnorm)[$(id)] * $(Vx) + $(ynorm)[$(id)] * $(Vy);

    const scalar g_est = -v_proj * $(Vt) / ($(b_reg) + v_proj * v_proj);

    $(V) += DT * (g_est - $(V)) / $(tau_m);
    """,
    extra_global_params=[("xnorm", "scalar*"), ("ynorm", "scalar*")],
    additional_input_vars=[
        ("Isyn_x", "scalar", 0.0),
        ("Isyn_y", "scalar", 0.0),
        ("Isyn_t", "scalar", 0.0),
    ],
    is_auto_refractory_required=False,
)

out_neuron = genn_model.create_custom_neuron_class(
    "OUT",
    param_names=["tau_m", "g_filt_bias", "g_filt_scale"],
    var_name_types=[("S_left", "scalar"), ("S_right", "scalar"), ("V_est", "scalar")],
    sim_code="""
   $(S_left) = $(Isyn_left);
   $(S_right) = $(Isyn_right);
   const scalar g_filt_left = 1./(1.+exp(-4.*($(S_left)-$(g_filt_bias))/$(g_filt_scale)));
   const scalar g_filt_right = 1./(1.+exp(-4.*($(S_right)-$(g_filt_bias))/$(g_filt_scale)));

   $(V_est) += DT * (g_filt_left * g_filt_right * 0.5 * ($(S_left) + $(S_right)) - $(V_est)) / $(tau_m);
   """,
    additional_input_vars=[("Isyn_left", "scalar", 0.0), ("Isyn_right", "scalar", 0.0)],
)

cont_wu = genn_model.create_custom_weight_update_class(
    "cont_wu",
    param_names=[],
    var_name_types=[("g", "scalar")],
    synapse_dynamics_code="$(addToInSyn, $(g) * $(V_pre));"
)
