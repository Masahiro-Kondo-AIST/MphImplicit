#include <cstdio>
#include <cstring>
#include "log.h"
#include "typedefs.h"

using namespace std;

static double ParticleDistance = -1.0;
static vec_t UpperDomain;
static vec_t LowerDomain;

static char boidname[256] = "sample.boid";
static char gridname[256] = "sample.grid";

struct Cuboid_t{
    double space;
    iType_t type;
    vec_t lower;
    vec_t upper;
    vec_t velocity;
};

typedef index_t<Cuboid_t>  iCub_t;
static vector_t<Cuboid_t,iCub_t> Cuboid;

static distinct_t Type;
static vPclData_t Position;
static vPclData_t Velocity;


static void readfile(const char *fname);
static int  readCuboid(FILE *fp,const char* endcommand);
static void genparticle();
static void writefile(const char *fname);

int main(int argc,char *argv[])
{
    if(argc>1){
        sprintf(boidname,"%s.boid",argv[1]);
        sprintf(gridname,"%s.grid",argv[1]);
    }
    
    readfile(boidname);
    genparticle();
    writefile(gridname);
}

void readfile(const char* fname)
{
    char buf[1024];
    char token[256];
    int pcldistflag=0, lowerflag=0, upperflag=0;
    FILE *fp = fopen(fname,"r");
    while(fp!=NULL && !feof(fp) && !ferror(fp)){
        if(fgets(buf,sizeof(buf),fp)==NULL)continue;
        if(buf[0]=='#')continue;
        if(sscanf(buf,"%s",token)!=1){fprintf(stderr,"token:%s\n",token);continue;}
    	//fprintf(stderr,"%s\n",token);
        if(strcmp(token,"ParticleDistance")==0){
            if(sscanf(buf," %*s %lf",&ParticleDistance)!=1){fprintf(stderr,"ParticleDistance count not 1\n");goto err;}
            pcldistflag=1;
        }
        if(strcmp(token,"LowerDomain")==0){
            if(sscanf(buf," %*s %lf %lf %lf",&LowerDomain[0], &LowerDomain[1], &LowerDomain[2])!=3){fprintf(stderr,"LowerDomain count not 3\n");goto err;}
            lowerflag=1;
        }
        if(strcmp(token,"UpperDomain")==0){
            if(sscanf(buf," %*s %lf %lf %lf",&UpperDomain[0], &UpperDomain[1], &UpperDomain[2])!=3){fprintf(stderr,"UpperDomain count not 3\n");goto err;}
            upperflag=1;
        }
        else if(strcmp(token,"StartCuboid")==0){
            if(readCuboid(fp,"EndCuboid")!=0)goto err;
        };
    }
    if(pcldistflag==0)fprintf(stderr,"no ParticleDistance");
    if(lowerflag==0)fprintf(stderr,"no LowerDomain");
    if(upperflag==0)fprintf(stderr,"no UpperDomain");
    return;
    err:
    fprintf(stderr,"error: \n\tfile:%s\n\tline:%s,token:%s\n",fname,buf,token);
    return;
}

int readCuboid(FILE *fp,const char *endcommand)
{
//    char buf[1024];
    char token[256];
    Cuboid_t cub;
    int spaceflag=0;
    int typeflag=0, lowerflag=0, upperflag=0, veloflag=0;
    cub.type = iType_t(0);
    cub.lower = vec_t(0.0,0.0,0.0);
    cub.upper = vec_t(0.0,0.0,0.0);
    cub.velocity = vec_t(0.0,0.0,0.0);
    while(1){
        fprintf(stderr, "line %d\n", __LINE__);
        if(fscanf(fp,"%s",token)!=1)continue;
        if(strcmp(token,endcommand)==0)break;
        else if(strcmp(token,"Spacing")==0){
            if(fscanf(fp, "%lf", &cub.space)!=1)goto err;
            spaceflag=1;
        }
        else if(strcmp(token,"Type")==0){
            if(fscanf(fp,"%d",&cub.type.setvalue())!=1)goto err;
            typeflag=1;
        }
        else if(strcmp(token,"Lower")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cub.lower[iDim])!=1)goto err;
            }
            lowerflag=1;
        }
        else if(strcmp(token,"Upper")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cub.upper[iDim])!=1)goto err;
            }
            upperflag=1;
        }
        else if(strcmp(token,"Velocity")==0){
            for(int iDim=0;iDim<3;++iDim){
                if(fscanf(fp,"%lf",&cub.velocity[iDim])!=1)goto err;
            }
            veloflag=1;
        }
        else{
            fprintf(stderr,"no such indication\n");
            goto err;
        }
    }

    if(spaceflag==0)fprintf(stderr,"no indecatio to Spacing");
    if(typeflag==0)fprintf(stderr,"no indecatio to Type");
    if(lowerflag==0)fprintf(stderr,"no indecatio to Lower");
    if(upperflag==0)fprintf(stderr,"no indecatio to Upper");
    if(veloflag==0)fprintf(stderr,"no indecatio to Velocity");
    if(!(spaceflag && typeflag && lowerflag && upperflag && veloflag))return 1;
        fprintf(stderr, "line %d\n", __LINE__);
    Cuboid += cub;
    return 0;
    
 err:
    fprintf(stderr,"error: token:%s",token);
    return 1;
}


            
void genparticle()
{
    const iCub_t cubcount = Cuboid.count();
    for(iCub_t iCub(0);iCub<cubcount;++iCub){
        const Cuboid_t& cub = Cuboid[iCub];
        const vec_t width = cub.upper-cub.lower;
        const vec3<int> count = vec3<int>(round(width[0]/cub.space),round(width[1]/cub.space),round(width[2]/cub.space));
        const vec_t spacing = vec_t(width[0]/count[0],width[1]/count[1],width[2]/count[2]);

        for(double px=cub.lower[0]+0.5*spacing[0]; px<cub.upper[0]-0.49*spacing[0]; px+=spacing[0]){
            for(double py=cub.lower[1]+0.5*spacing[1]; py<cub.upper[1]-0.49*spacing[1]; py+=spacing[1]){
                 for(double pz=cub.lower[2]+0.5*spacing[2]; pz<cub.upper[2]-0.49*spacing[2]; pz+=spacing[2]){

                    // fprintf(stderr, "p %e %e %e\n", px, py, pz);
                    Type += cub.type;
                    Position += vec_t(px,py,pz);
                    Velocity += cub.velocity;
                }
            }
        }
    }
    fprintf(stderr, "%d particles were generated\n", Type.count().getvalue());

}



void writefile(const char *fname)
{
    FILE *fp = fopen(fname,"w");
    const iPcl_t& pclCount = Type.count();
    fprintf(fp,"%lf\n",0.0);
    fprintf(fp,"%d %e  %e %e %e  %e %e %e\n",
            pclCount.getvalue(),
            ParticleDistance,
            LowerDomain[0], UpperDomain[0],
            LowerDomain[1], UpperDomain[1],
            LowerDomain[2], UpperDomain[2]
            );
    for(iPcl_t iPcl(0);iPcl<pclCount;++iPcl){
        fprintf(fp,"%d   %e %e %e    %e %e %e\n",
                Type[iPcl].getvalue(),
                Position[iPcl][0],Position[iPcl][1],Position[iPcl][2],
                Velocity[iPcl][0],Velocity[iPcl][1],Velocity[iPcl][2]
                );
    }
    fclose(fp);
}

                
