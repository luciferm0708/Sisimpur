from django.test import SimpleTestCase
from django.urls import reverse, resolve
from frontend.views import index


class UrlsTest(SimpleTestCase):
    """Test cases for the frontend app URLs"""

    def test_index_url_resolves(self):
        """Test that the index URL resolves to the correct view function"""
        url = reverse('index')
        self.assertEqual(resolve(url).func, index)

    def test_index_url_name(self):
        """Test that the index URL name resolves to the correct path"""
        url = reverse('index')
        self.assertEqual(url, '/')
