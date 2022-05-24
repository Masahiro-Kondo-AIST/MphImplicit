#ifndef __TYPEDEFS_H__INCLUDED__
#define __TYPEDEFS_H__INCLUDED__

#include <cstdio>
#include <vector>
#include <cstdarg>
#include "vec3T.hpp"

#define GHOST -1

template <typename T>
class index_t{
private:
  int i;
public: 
  index_t():i(0){}
  index_t(const index_t& index):i(index.i){}
  explicit index_t(const int index):i(index){}
  index_t& operator= (const index_t& index){i =index.i;return *this;}
  index_t& operator+=(const index_t& index){i+=index.i;return *this;}
  bool operator==(const index_t& index)const{return i==index.i;}
  bool operator!=(const index_t& index)const{return i!=index.i;}
  bool operator< (const index_t& index)const{return i< index.i;}
  bool operator<=(const index_t& index)const{return i<=index.i;}
  bool operator> (const index_t& index)const{return i>=index.i;}
  bool operator>=(const index_t& index)const{return i> index.i;}
  index_t& operator++(){++i;return *this;}
  index_t& operator--(){--i;return *this;}
  int getvalue()const{return i;}
  int& setvalue(){return i;}
  T gettype()const{return T();}
};

struct PCL_t{};
struct TYPE_t{};
struct TABLE_t{};
struct TYPELIST_t{};

typedef index_t<PCL_t> iPcl_t;
typedef index_t<TYPE_t> iType_t;
typedef index_t<TABLE_t> iTable_t;
typedef index_t<TYPELIST_t> iTypelist_t;

template<typename T,typename T2>
class vector_t{
protected:
    T2 valuecount;
    std::vector<T> value;
public:
  vector_t(){}
  vector_t(const vector_t& vec):value(vec.value){}
  explicit vector_t(const std::vector<T>& vec):value(vec){}
  explicit vector_t(const T& i){push_back(i);}
  T& operator[](const T2& i){return value[i.getvalue()];}
  const T& operator[](const T2& i)const{return value[i.getvalue()];}
  size_t size()const{return value.size();}
  const T2& count()const{return valuecount;}
  void resize(const T2& i){value.resize(i.getvalue());valuecount=i;}
  vector_t& operator+=(const T& i){value.push_back(i);++valuecount;return *this;}
  vector_t& operator+=(const vector_t& list){
    const T2& count = list.count();
    for(T2 i=T2(0);i<count;++i){value.push_back(list[i]);}
    valuecount += count;
    return *this;
  }
  vector_t operator+(const T& i){return vector_t(*this)+=i;}
  vector_t operator+(const vector_t& list){return vector_t(*this)+=list;}
  
};

typedef vector_t<iType_t,iPcl_t>         distinct_t;
typedef vector_t<iPcl_t,iTable_t>        table_t;
typedef vector_t<iType_t,iTypelist_t>    typelist_t; //typelist_t()+FluidType+WallType
typedef vector_t<iTable_t,iPcl_t>        invtable_t;
typedef vector_t<iTypelist_t,iType_t>    invtypelist_t;


table_t find(const distinct_t& distinct,const iType_t type);
invtable_t createInverseTable(const table_t& table,const iPcl_t& pclCount);
invtypelist_t createInverseTypelist(const typelist_t& typelist,const iType_t& typeCount);

typedef vec3<double> vec_t;
typedef mat3<double> mat_t;

typedef vector_t<double,iPcl_t>          dPclData_t;
typedef vector_t<vec_t ,iPcl_t>          vPclData_t;
typedef vector_t<mat_t ,iPcl_t>          mPclData_t;
typedef vector_t<int   ,iType_t>         iTypeData_t;
typedef vector_t<double,iType_t>         dTypeData_t;
typedef vector_t<vec_t ,iType_t>         vTypeData_t;
typedef vector_t<mat_t ,iType_t>         mTypeData_t;

typedef vector_t<double,iTable_t>        dTableData_t;
typedef vector_t<vec_t ,iTable_t>        vTableData_t;
typedef vector_t<mat_t ,iTable_t>        mTableData_t;
typedef vector_t<double,iTypelist_t>     dTypelistData_t;
typedef vector_t<vec_t ,iTypelist_t>     vTypelistData_t;
typedef vector_t<mat_t ,iTypelist_t>     mTypelistData_t;




//////////////////////////////////////////////////
typedef vec3<double> vec_t;
typedef mat3<double> mat_t;



#endif// __TYPEDEFS_H__INCLUDED__

