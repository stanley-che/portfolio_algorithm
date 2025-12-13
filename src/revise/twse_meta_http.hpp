//twse_meta_http.hpp
#pragma once
#include <string>
#include <curl/curl.h>

namespace meta {

inline size_t twse_write_cb(void* c, size_t s, size_t n, void* outp){
    static_cast<std::string*>(outp)->append(static_cast<char*>(c), s*n);
    return s*n;
}

inline bool TwseMetaBuilder::http_get_json(const std::string& url,
                                          std::string& body,
                                          long connect_timeout_sec,
                                          long timeout_sec) {
    body.clear();
    CURL* h = curl_easy_init();
    if(!h) return false;

    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "User-Agent: Mozilla/5.0");
    headers = curl_slist_append(headers, "Accept: application/json,text/plain,*/*");
    headers = curl_slist_append(headers, "Referer: https://www.twse.com.tw/");

    curl_easy_setopt(h, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_ACCEPT_ENCODING, ""); // gzip/deflate
    curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, twse_write_cb);
    curl_easy_setopt(h, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, connect_timeout_sec);
    curl_easy_setopt(h, CURLOPT_TIMEOUT, timeout_sec);

    CURLcode rc = curl_easy_perform(h);

    curl_slist_free_all(headers);
    curl_easy_cleanup(h);

    return rc == CURLE_OK;
}

} // namespace twse
