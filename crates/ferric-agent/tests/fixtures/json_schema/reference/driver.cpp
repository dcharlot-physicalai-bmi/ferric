// Reference driver: llama.cpp's json_schema_to_grammar(schema, force_gbnf=true) over a JSON array of schema
// texts (stdin), printing a JSON array of {"ok": grammar} / {"err": message}.
#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <iterator>
int main() {
    std::string in((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    auto texts = nlohmann::json::parse(in);
    nlohmann::json out = nlohmann::json::array();
    for (const auto & t : texts) {
        nlohmann::json r;
        try {
            r["ok"] = json_schema_to_grammar(common_json::parse(t.get<std::string>()), true);
        } catch (const std::exception & e) {
            r["err"] = e.what();
        }
        out.push_back(r);
    }
    std::cout << out.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << std::endl;
}
