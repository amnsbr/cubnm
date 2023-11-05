#ifndef HELPERS_HPP
#define HELPERS_HPP

extern float get_env_or_default(std::string key, double value_default = 0.0);
extern int get_env_or_default(std::string key, int value_default = 0);
extern std::string extract_substring(const std::string& filePath, const std::string& delimiter1, const std::string& delimiter2);

#endif