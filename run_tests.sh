#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run all tests with verbose output
python manage.py test apps/frontend/tests apps/authentication/tests -v 2

# Run specific test modules if needed
# python manage.py test apps.frontend.tests.test_views
# python manage.py test apps.authentication.tests.test_models

# Run with coverage (if installed)
# coverage run --source='apps' manage.py test apps.frontend.tests apps.authentication.tests
# coverage report
# coverage html  # generates htmlcov/coming_soon.html
