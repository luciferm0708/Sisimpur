# SISIMPUR Project Tests

This document explains how to run the tests for the SISIMPUR project.

## Test Structure

The tests are organized by app and test type:

```
apps/
  ├── frontend/
  │   └── tests/
  │       ├── __init__.py
  │       ├── test_views.py
  │       ├── test_urls.py
  │       └── test_templates.py
  └── authentication/
      └── tests/
          ├── __init__.py
          ├── test_views.py
          └── test_models.py
```

## Running Tests

### Using the Django test runner

To run all tests:

```bash
python manage.py test
```

To run tests for a specific app:

```bash
python manage.py test apps.frontend
python manage.py test apps.authentication
```

To run a specific test module:

```bash
python manage.py test apps.frontend.tests.test_views
python manage.py test apps.authentication.tests.test_models
```

To run a specific test class:

```bash
python manage.py test apps.frontend.tests.test_views.IndexViewTest
```

To run a specific test method:

```bash
python manage.py test apps.frontend.tests.test_views.IndexViewTest.test_index_view_status_code
```

### Using the provided script

For convenience, a script is provided to run all tests:

```bash
./run_tests.sh
```

## Test Coverage

To measure test coverage, install the `coverage` package:

```bash
pip install coverage
```

Then run the tests with coverage:

```bash
coverage run --source='apps' manage.py test
coverage report
coverage html  # generates htmlcov/index.html
```

## Writing New Tests

When writing new tests:

1. Place them in the appropriate app's `tests` directory
2. Name test files with a `test_` prefix
3. Name test classes with a `Test` suffix
4. Name test methods with a `test_` prefix
5. Include docstrings to explain what each test is checking

Example:

```python
from django.test import TestCase

class MyFeatureTest(TestCase):
    """Tests for MyFeature"""
    
    def test_my_feature_works(self):
        """Test that MyFeature works as expected"""
        # Test code here
```
