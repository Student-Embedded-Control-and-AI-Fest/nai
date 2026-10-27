#include "../src/nai_motion_lpf5.h"
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
int main(int argc,char**argv){
    if(argc!=2) return 2;
    std::ifstream f(argv[1]);
    std::vector<float> raw,reference;
    float v[12];int n=0;
    while(f>>v[0]){
       for(int k=1;k<12;++k)if(!(f>>v[k]))return 4;
       for(int k=0;k<6;++k)raw.push_back(v[k]);
       for(int k=6;k<12;++k)reference.push_back(v[k]);
       ++n;
    }
    assert(n==90);
    nai_filter_raw6_lpf5(raw.data(),static_cast<uint16_t>(n),50,true,true);
    for(size_t i=0;i<raw.size();++i)
      if(std::fabs(raw[i]-reference[i])>2.e-6f){
        std::cerr<<"LPF mismatch "<<i<<" "<<raw[i]<<" vs "<<reference[i]<<"\n";return 5;
      }
    float constant[30]={};for(int t=0;t<5;t++)constant[t*6+2]=1.0f;
    nai_filter_raw6_lpf5(constant,5,50,true,true);
    for(int t=0;t<5;t++)assert(std::fabs(constant[t*6+2]-1.0f)<1e-6f);
    float feature[4];nai_accel_direction_magnitude(0,0,1,feature);
    assert(std::fabs(feature[2]-1.f)<1.e-6f&&std::fabs(feature[3]-1.f)<1.e-6f);
    nai_accel_direction_magnitude(0,0,0,feature);
    for(float x:feature)assert(x==0);
    std::cout<<"PASS: 5Hz C++ LPF agrees with JavaScript on "<<n<<" 6-axis samples; 1g and zero handling verified\n";
}
