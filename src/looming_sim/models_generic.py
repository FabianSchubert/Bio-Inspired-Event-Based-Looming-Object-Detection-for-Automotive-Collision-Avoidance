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

# LIF neuron model
lif_neuron = genn_model.create_custom_neuron_class(
    "LIF",
    param_names=["tau_m", "V_thresh", "V_reset"],
    var_name_types=[("V", "scalar"), ("VI", "scalar")],
    sim_code="""
    $(VI) = $(Isyn);
    $(V) += DT * ($(VI)-$(V))/$(tau_m);  // linear Euler
    """,
    threshold_condition_code="""
    $(V) >= $(V_thresh)
    """,
    reset_code="""
    $(V) = $(V_reset);  // reset to $(V_reset)
    """,
    is_auto_refractory_required=False,
)