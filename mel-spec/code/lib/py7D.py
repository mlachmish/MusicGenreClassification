import httplib2
import os
import urllib
import re
import urlparse
import xmltodict
import collections

import oauth7digital as oa7D
import api_settings

__name__ = 'py7D'
__doc__ = 'A lightweight python interface to 7Digital web service'
__author__ = 'Jason Rubenstein'
__version__ = '0.2'
__maintainer__ = 'Jason Rubenstein'
__email__ = 'jasondrubenstein@gmail.com'
__status__ = 'Beta'

SERVER = 'api.7digital.com'
API_VERSION = '1.2'
API_URL = 'https://%s/%s' % (SERVER, API_VERSION)
CLIP_URL = 'http://previews.7digital.com/clip/%s?' % (
                                                "%s")

class APIServiceException(Exception):
    pass


class APIClientException(Exception):
    pass


def _assemble_url(host, method, function, oauth, **kwargs):
    data = []

    for k,v in kwargs.iteritems():
        if isinstance(v, (int, float, long)):
            kwargs[k] = str(v)

    quote = lambda _param_name: urllib.quote_plus(
                        _param_name.replace('&amp;', '&').encode('utf8')
    )

    for name in kwargs.keys():
        data.append('='.join((name, quote(kwargs[name]))))

    data = '&'.join(data)

    url = os.path.join(host, method)
    if function:
        url = os.path.join(url, function)

    if not oauth:
        url = "%s%s" % (url, '?oauth_consumer_key=%s&country=%s' % (
                                                        api_settings.oauthkey,
                                                        api_settings.country)
                       )
    if data:
        url += "&%s" % data

    return url


def _execute(method, function, access_token=None, **kwargs):

    ''' This common execution function is overloaded to
        handle standard requests, 2-legged oauth requests, and
        3-legged oauth requests.
    '''
    dc = kwargs.pop('dict_constructor', None) or collections.OrderedDict
    is_2legged = kwargs.pop('is_2legged', False)
    http_method = kwargs.pop('http_method', "GET")
    body = kwargs.pop('body', '')

    oauth = False
    if access_token or is_2legged:
        oauth = True

    url = _assemble_url(API_URL, method, function, oauth, **kwargs)

    # a request is one of 3legged oauth, a 2legged oauth, or standard
    if access_token:
        http_response, content = oa7D.request_3legged(url,
                                                      access_token,
                                                      http_method=http_method,
                                                      body=body)
        api_response = xmltodict.parse(
            content, xml_attribs=True, dict_constructor=dc)
    else:
        if is_2legged is True:
            http_response, content = oa7D.request_2legged(
                                                    url,
                                                    http_method=http_method)
        else:
            http_response, content = httplib2.Http().request(url)

        api_response = xmltodict.parse(
            content, xml_attribs=True, dict_constructor=dc)

    if api_response['response']['@status'] == "error":
        raise APIServiceException('Error code %s: %s' % (
                api_response['response']['error']['@code'],
                api_response['response']['error']['errorMessage']
                )
        )

    api_response['http_headers'] = http_response
    return api_response


def request(method, function, **kwargs):
    ''' Input:
            method      : a valid 7Digital method
            function    : a valid function for the method
            Other kwargs specific to the API method/function

        Output:
            A python Ordered Dictionary of the results of the
            API, converted from XML.
    '''
    if kwargs.get('access_token'):
        raise APIClientException(
            "Please use oauth_request() for calls containing access_token")

    return _execute(method, function, **kwargs)


def oauth_3legged_request(method, function, access_token, **kwargs):
    ''' Input:
            method      : a valid 7Digital method for 3-legged oauth
            function    : a valid function for the method
            access_token: the access_token required for 3-legged oauth
            Other kwargs specific to the API method/function

        Output:
            A python Ordered Dictionary of the results of the
            API, converted from XML.
    '''
    if not access_token:
        raise APIClientException("access_token required for oauth request")

    return _execute(method, function, access_token, **kwargs)


def oauth_2legged_request(method, function, **kwargs):
    ''' Input:
            method      : a valid 7Digital method; commonly 'oauth'
            function    : a valid function for the method
            Other kwargs specific to the API method/function

        Output:
            A python Ordered Dictionary of the results of the
            API, converted from XML.
    '''
    return _execute(method, function, is_2legged=True, **kwargs)



def preview_clip(track_id):
    ''' return a preview clip in mp3 format,
        from the 7digital preview servers.

        Input:
            valid track_id

        Output:
            An mp3 (attachment) preview-clip for the requested track.
    '''
    response, content = oa7D.request_2legged(
            CLIP_URL % (track_id)
    )
    return response, content


def preview_url(track_id):
    ''' return the URL for preview clip, from the 7digital preview servers.

        Input:
            valid track_id

        Output:
            The OAuth-signed URL for a preview clip. This URL can be used
            to retrieve the mp3 preview-clip from 7digital.
    '''
    s_req = oa7D.signed_request(CLIP_URL % (track_id))
    return s_req.to_url()
