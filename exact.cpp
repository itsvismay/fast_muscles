#include <igl/opengl/glfw/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/writeDMAT.h>
#include <igl/jet.h>
#include <igl/png/readPNG.h>
#include <igl/png/writePNG.h>
#include <imgui/imgui.h>
#include <json.hpp>
// #include <LBFGS.h>


#include <sstream>
#include <iomanip>
// #include <omp.h>

#include "exact/setup_store.h"


using namespace Eigen;
using namespace std;
using json = nlohmann::json;
// using namespace LBFGSpp;

using Store = exact::Store;
json j_input;


int main(int argc, char *argv[])
{
  int fancy_data_index,debug_data_index,discontinuous_data_index;
  std::cout<<"-----Configs-------"<<std::endl;
    std::string inputfile;
    int num_threads = 1;
    if(argc==3){
      num_threads = std::stoi(argv[2]);
      omp_set_num_threads(num_threads);
      std::ifstream input_file(argv[1]);
      input_file >> j_input;
      std::cout<<"Threads: "<<Eigen::nbThreads( )<<std::endl;
    }else{
      cout<<"Run as: ./famu input.json <threads>"<<endl;
      exit(0);
    }
    Eigen::initParallel();

    exact::Store store;
    store.jinput = j_input;
    exact::setupStore(store);
	
		
	

}
