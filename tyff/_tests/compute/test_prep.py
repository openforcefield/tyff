from tyff.compute.prep import compute_configs_from_data_entries, get_liquid_deduplication_key


def test_density_and_dielectric_produce_same_compute_config(density_entry, dielectric_entry):
    """Density and dielectric targets should produce identical configs: pure bulk liquids."""
    compute_configs = compute_configs_from_data_entries(
        data_entries=[density_entry, dielectric_entry],
        force_field="test-10.0",
        n_molecules=600,
    )

    # for each step
    assert len(compute_configs) == 6

    for this_config in compute_configs:
        # these should be deduplicated
        assert len(this_config) == 1

    assert len(set(get_liquid_deduplication_key(val[0]) for val in compute_configs)) == 1


def test_basic_deduplciation(density_entry):
    """
    When data entry, force field, and n_molecules are the same, the resulting compute
    configs should "deduplicate" out, in the form of get_liquid_deduplication_key()
    returning identical values.
    """

    compute_configs = [
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="test-0.0a",
            n_molecules=600,
        ),
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="test-0.0a",
            n_molecules=600,
        ),
    ]

    assert len(set(get_liquid_deduplication_key(val[0]) for val in compute_configs)) == 1


def test_different_force_fields_do_not_deduplicate(density_entry):

    compute_configs = [
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="test-10.0",
            n_molecules=600,
        ),
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="test-10.0.1",
            n_molecules=600,
        ),
    ]

    assert len(set(get_liquid_deduplication_key(val[0]) for val in compute_configs)) == 2


def test_different_n_molecules_do_not_deduplicate(density_entry):

    compute_configs = [
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="foo-bar.x",
            n_molecules=1000,
        ),
        *compute_configs_from_data_entries(
            data_entries=[density_entry],
            force_field="foo-bar.x",
            n_molecules=1001,
        ),
    ]

    assert len(set(get_liquid_deduplication_key(val[0]) for val in compute_configs)) == 2
