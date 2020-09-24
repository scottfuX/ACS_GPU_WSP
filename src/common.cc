#include "common.h"
#include <ctime>
#include <cstdio>
#include <iostream>
#include <fstream>

std::map<std::string, pj> global_record;


void record(std::string key, int64_t value) {
    global_record[key] = pj(value);
}

void record(std::string key, double value) {
    global_record[key] = pj(value);
}


void record(std::string key, const std::string value) {
    global_record[key] = pj(value);
}


void record(std::string key, const std::vector<int> &values) {
    std::vector<pj> pj_vec;
    for (auto e : values) {
        pj_vec.push_back( pj((int64_t)e) );
    }
    global_record[key] = pj(pj_vec);
}

void record(std::string key, const std::vector<float> &values) {
    std::vector<pj> pj_vec;
    for (auto e : values) {
        pj_vec.push_back( pj((float)e) );
    }
    global_record[key] = pj(pj_vec);
}

void record(std::string key, const std::vector<double> &values) {
    std::vector<pj> pj_vec;
    for (auto e : values) {
        pj_vec.push_back( pj((double)e) );
    }
    global_record[key] = pj(pj_vec);
}


void record(std::string key, const std::vector<pj> &values) {
    global_record[key] = pj(values);
}


void record(std::string key, const std::map<std::string, pj> &dict) {
    global_record[key] = pj(dict);
}


std::string global_record_to_string() {
    return pj(global_record).serialize( /* prettify = */ true );
}


// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string get_current_date_time(std::string format) {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), format.c_str(), &tstruct);

    return buf;
}




int bupt_outfile_app(std::string filename,std::vector<float> &vec)
{
    std::ofstream fout;
    fout.open(filename.c_str(),std::ios::app);
    if(!fout)
    {
        std::cout<<"文件打开失败!"<<std::endl;
        return -1;
    }
    for(uint32_t i=0;i<vec.size();i++)
    {
        fout << std::to_string(vec[i]) << " ";
    }
    fout.close();

    return 0;
}

int bupt_outfile_app(std::string filename,std::vector<double> &vec)
{
    std::ofstream fout;
    fout.open(filename.c_str(),std::ios::app);
    if(!fout)
    {
        std::cout<<"文件打开失败!"<<std::endl;
        return -1;
    }
    for(uint32_t i=0;i<vec.size();i++)
    {
        fout << std::to_string(vec[i]) << " ";
    }
    fout.close();

    return 0;
}

void bupt_chk_and_del(std::string str_title){
       std::ifstream fin(str_title);
       if(fin)
       {
            fin.close();
            if(remove(str_title.c_str()))
                std::cout<<"文件删除失败!"<<std::endl;
       }
}

int bupt_outfile_app_newline(std::string filename,std::string str){
    std::ofstream fout(filename.c_str(),std::ios::app);
    if(!fout)
    {
        std::cout<<"文件打开失败!"<<std::endl;
        return -1;
    }
    fout << std::endl << str << std::endl;
    fout.close();
    return 0;
}

int bupt_outfile_app_newline(std::string filename,const char* str){
    std::ofstream fout(filename.c_str(),std::ios::app);
    if(!fout)
    {
        std::cout<<"文件打开失败!"<<std::endl;
        return -1;
    }
    fout << std::endl << str << std::endl;
    fout.close();
    return 0;
}

int bupt_outfile_app_line(std::string filename,std::string str){
    std::ofstream fout(filename.c_str(),std::ios::app);
    if(!fout)
    {
        std::cout<<"文件打开失败!"<<std::endl;
        return -1;
    }
    fout <<  str ;
    fout.close();
    return 0;
}



int bupt_outfile_out(std::string filename,std::vector<float> &vec)
{
    std::ofstream fout(filename.c_str(),std::ios::out);
    if(!fout)
    {
        std::cout<<"文件打开失败!"<<std::endl;
        return -1;
    }
    for(uint32_t i=0;i<vec.size();i++)
    {
        fout<<std::to_string(vec[i])<<" ";
    }
    fout.close();
    return 0;
}

