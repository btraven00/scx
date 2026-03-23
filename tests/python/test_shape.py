from conftest import EXPECTED_N_OBS, EXPECTED_N_VARS


def test_shape_matches_reference(scx_adata, ref_adata):
    assert scx_adata.shape == ref_adata.shape


def test_n_obs(scx_adata):
    assert scx_adata.n_obs == EXPECTED_N_OBS


def test_n_vars(scx_adata):
    assert scx_adata.n_vars == EXPECTED_N_VARS


def test_obs_names_match(scx_adata, ref_adata):
    assert list(scx_adata.obs_names) == list(ref_adata.obs_names)


def test_var_names_match(scx_adata, ref_adata):
    assert list(scx_adata.var_names) == list(ref_adata.var_names)
