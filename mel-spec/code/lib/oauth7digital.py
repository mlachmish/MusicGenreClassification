import httplib
import re
import sys

import api_settings
import oauth2 as oauth

SERVER = 'api.7digital.com'
API_VERSION = '1.2'
REQUEST_TOKEN_URL = 'https://%s/%s/oauth/requesttoken' % (SERVER, API_VERSION)
ACCESS_TOKEN_URL = 'https://%s/%s/oauth/accesstoken' % (SERVER, API_VERSION)
AUTHORIZATION_URL = 'https://account.7digital.com/%s/oauth/authorise'


def _consumer():
    return oauth.Consumer(api_settings.oauthkey, api_settings.secret)


def _token_from_response_content(content):

    try:
        key = re.search(
            "<oauth_token>(\w.+)</oauth_token>",
            content).groups()[0]
        secret = re.search(
            "<oauth_token_secret>(\w.+)</oauth_token_secret>",
            content).groups()[0]
    except AttributeError, e:
        return "Error processing response from 7digital: (%s) [AttributeError: %s]" % (content, e)

    return oauth.Token(key, secret)


def request_2legged(url, http_method="GET"):
    client = oauth.Client(_consumer())
    response, content = client.request(
        url,
        headers = {"Content-Type":"application/x-www-form-urlencoded"},
        body="country=%s" % api_settings.country,
        method = http_method
    )
    return response, content


def request_token():
    response, content = request_2legged(REQUEST_TOKEN_URL)

    if response['status'] == '200':
        return _token_from_response_content(content)

    return response, content


def authorize_request_token(token, debug=False):
    keyed_auth_url = AUTHORIZATION_URL % api_settings.oauthkey
    auth_url="%s?oauth_token=%s" % (keyed_auth_url, token.key)

    if debug:
        print 'Authorization URL: %s' % auth_url
        oauth_verifier = raw_input(
            'Please go to the above URL and authorize the app. \
            Hit return when you have been authorized: '
        )

    return auth_url


def request_access_token(token):
    client = oauth.Client(_consumer(), token=token)
    response, content = client.request(
            ACCESS_TOKEN_URL,
            headers={"Content-Type":"application/x-www-form-urlencoded"}
    )

    return _token_from_response_content(content)


def request_3legged(url, access_token, http_method="GET", body=''):
    ''' Once you have an access_token authorized by a customer,
        execute a request on their behalf
    '''
    client = oauth.Client(_consumer(), token=access_token)
    response = client.request(
            url,
            headers={"Content-Type":"application/x-www-form-urlencoded"},
            method=http_method,
            body=body
    )

    return response

def signed_request(url, access_token=None):
    ''' return a signed request. Usually not used, save for specific
        functions like deriving a preview-clip URL.
    '''
    consumer = _consumer()

    req = oauth.Request.from_consumer_and_token(
                consumer,
                http_url=url,
                is_form_encoded=True,
                parameters={'country':api_settings.country})

    signing_method = oauth.SignatureMethod_HMAC_SHA1()
    req.sign_request(signing_method, consumer, access_token)

    return req
