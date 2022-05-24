#include "typedefs.h"

table_t find(const distinct_t& distinct,const iType_t type)
{
  table_t returnTable;
  returnTable.resize(iTable_t(0));
  const iPcl_t& pclCount = distinct.count();
  for(iPcl_t iPcl(0);iPcl<pclCount;++iPcl){
    if(distinct[iPcl]==type)returnTable+=iPcl;
  }
  return returnTable;
}

invtable_t createInverseTable(const table_t& table,const iPcl_t& pclCount)
{
  invtable_t reval;
  reval.resize(pclCount);
  for(iPcl_t iPcl(0);iPcl<pclCount;++iPcl){
    reval[iPcl]=iTable_t(-1);
  }
  const iTable_t& tableCount = table.count();
  for(iTable_t iTable(0);iTable<tableCount;++iTable){
    reval[table[iTable]]=iTable;
  }
  return reval;
}

invtypelist_t createInverseTypelist(const typelist_t& typelist,const iType_t& typeCount)
{
  invtypelist_t reval;
  reval.resize(typeCount);
  for(iType_t iType(0);iType<typeCount;++iType){
    reval[iType]=iTypelist_t(-1);
  }
  const iTypelist_t& typelistCount = typelist.count();
  for(iTypelist_t iTypelist(0);iTypelist<typelistCount;++iTypelist){
    reval[typelist[iTypelist]]=iTypelist;
  }
  return reval;
}










