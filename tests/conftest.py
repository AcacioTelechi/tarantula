import pytest

@pytest.fixture
def fixtures_dir(request):
    return request.config.rootpath / "tests" / "fixtures"
