from tools import phase_c_feasibility as gate


def test_analytical_schedule_counts_eos_and_windows():
    result = gate.analytical_schedule([[1, 2], list(range(128))])
    assert result["training_steps"] == 3
    assert result["training_target_tokens"] == 132
    assert result["training_sequence_length_squared_sum"] == 3 * 3 + 128 * 128 + 1
    assert result["actual_analytical_flops"] == (
        result["core_analytical_flops"] + result["output_projection_flops"]
    )


def test_modal_cost_components_are_explicit():
    components, total = gate.hourly_cost()
    assert set(components) == {"l4_gpu", "four_cpu_cores", "sixteen_gib_memory"}
    assert total == sum(components.values())
    assert total > components["l4_gpu"]
