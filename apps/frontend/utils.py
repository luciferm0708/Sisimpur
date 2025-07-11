import mailchimp_marketing as MailchimpMarketing
from mailchimp_marketing.api_client import ApiClientError
import logging
import json
import requests

logger = logging.getLogger(__name__)


class EmailValidationService:
    def __init__(self, api_key_01, api_key_02):
        self.api_key_01 = api_key_01
        self.api_key_02 = api_key_02

    def is_valid_check_01(self, email):
        url = "http://apilayer.net/api/check"
        params = {
            "access_key": self.api_key_01,
            "email": email,
            "smtp": 1,
            "format": 1,
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            return data.get("smtp_check", False)
        except requests.RequestException:
            return False

    def is_valid_check_02(self, email):
        url = "https://api.emailvalidation.io/v1/info"
        params = {
            "apikey": self.api_key_02,
            "email": email,
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            return data.get("smtp_check", False)
        except requests.RequestException:
            return False


class MailchimpService:
    def __init__(self, api_key, server_prefix):
        """
        Initialize the Mailchimp service with API key and server prefix

        Args:
            api_key (str): Mailchimp API key
            server_prefix (str): Server prefix from API key (e.g., 'us13')
        """
        self.client = MailchimpMarketing.Client()
        self.client.set_config({"api_key": api_key, "server": server_prefix})

    def ping(self):
        """Test the connection to Mailchimp API"""
        try:
            response = self.client.ping.get()
            return response
        except ApiClientError as error:
            logger.error(f"Mailchimp API error: {error}")
            return None

    def add_subscriber(self, list_id, email, status="pending"):
        """
        Add a subscriber to a Mailchimp list

        Args:
            list_id (str): Mailchimp list ID
            email (str): Subscriber's email address
            status (str): Subscription status ('subscribed', 'pending', 'unsubscribed')
                          Default is 'pending' which requires confirmation

        Returns:
            dict: Response from Mailchimp API or error message
        """
        try:
            response = self.client.lists.add_list_member(
                list_id,
                {
                    "email_address": email,
                    "status": status,
                },
            )
            return {"success": True, "data": response}
        except ApiClientError as error:
            error_json = json.loads(error.text)
            # Check if it's already subscribed error
            if error_json.get("title") == "Member Exists":
                return {"success": False, "error": "This email is already subscribed."}
            logger.error(f"Mailchimp API error: {error}")
            return {"success": False, "error": str(error)}
