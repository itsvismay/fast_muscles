
#ifndef READ_CONFIG_FILES 
#define READ_CONFIG_FILES

#include <Eigen/Dense>
#include <iostream>
#include <json.hpp>
#include "store.h"
using Store = famu::Store;


namespace famu
{

	void read_config_files(Store& store);
}

#endif