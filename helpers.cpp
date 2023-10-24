#ifndef HELPERS
#define HELPERS

float get_env_or_default(std::string key, double value_default = 0.0) {
    const char* value = std::getenv(key.c_str());
    if (value != nullptr) {
        return atof(value);
    } else {
        return value_default;
    }
}

int get_env_or_default(std::string key, int value_default = 0) {
    const char* value = std::getenv(key.c_str());
    if (value != nullptr) {
        return atoi(value);
    } else {
        return value_default;
    }
}

std::string extract_substring(const std::string& filePath, const std::string& delimiter1, const std::string& delimiter2)
{
    size_t startPos = filePath.find(delimiter1);
    if (startPos == std::string::npos)
        return "";

    startPos += delimiter1.length();

    size_t endPos = filePath.find(delimiter2, startPos);
    if (endPos == std::string::npos)
        return "";

    return filePath.substr(startPos, endPos - startPos);
}

#endif