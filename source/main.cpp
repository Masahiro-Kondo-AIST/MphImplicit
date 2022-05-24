//================================================================================================//
//------------------------------------------------------------------------------------------------//
//    MPH-I : Moving Particle Hydrodynamics for Incompressible Flows (implicit calculation)       //
//------------------------------------------------------------------------------------------------//
//    Developed by    : Masahiro Kondo                                                            //
//    Distributed in  : 2022                                                                      //
//    Lisence         : GPLv3                                                                     //
//    For instruction : see README                                                                //
//    For theory      : see the following references                                              //
//     [1] CPM 8 (2021) 69-86,       https://doi.org/10.1007/s40571-020-00313-w                   //
//     [2] JSCES Paper No.20210006,  https://doi.org/10.11421/jsces.2021.20210006                 //
//     [3] CPM 9 (2022) 265-276,     https://doi.org/10.1007/s40571-021-00408-y                   //
//     [4] CMAME 385 (2021) 114072,  https://doi.org/10.1016/j.cma.2021.114072                    //
//     [5] JSCES Paper No.20210016,  https://doi.org/10.11421/jsces.2021.20210016                 //
//    Copyright (c) 2022                                                                          //
//    Masahiro Kondo & National Institute of Advanced Industrial Science and Technology (AIST)    //
//================================================================================================//


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <assert.h>
#include <omp.h>

#include "errorfunc.h"
#include "log.h"

const double DOUBLE_ZERO[32]={0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};

using namespace std;
#define TWO_DIMENSIONAL

#define DIM 3

// Property definition
#define TYPE_COUNT   4
#define FLUID_BEGIN  0
#define FLUID_END    2
#define WALL_BEGIN   2
#define WALL_END     4

#define  DEFAULT_LOG  "sample.log"
#define  DEFAULT_DATA "sample.data"
#define  DEFAULT_GRID "sample.grid"
#define  DEFAULT_PROF "sample%03d.prof"
#define  DEFAULT_VTK  "sample%03d.vtk"

// Calculation and Output
static double ParticleSpacing=0.0;
static double ParticleVolume=0.0;
static double DomainMin[DIM];
static double DomainMax[DIM];
static double OutputInterval=0.0;
static double OutputNext=0.0;
static double VtkOutputInterval=0.0;
static double VtkOutputNext=0.0;
static double EndTime=0.0;
static double Time=0.0;
static double Dt=1.0e100;
static double DomainWidth[DIM];
#pragma acc declare create(ParticleSpacing,ParticleVolume,Dt,DomainMin,DomainMax,DomainWidth)

#define Mod(x,w) ((x)-(w)*floor((x)/(w)))   // mod 


#define MAX_NEIGHBOR_COUNT 512
// Particle
static int ParticleCount;
static int *ParticleIndex;                // original particle id
static int *Property;                     // particle type
static double (*Mass);                    // mass
static double (*Position)[DIM];           // coordinate
static double (*Velocity)[DIM];           // momentum
static double (*Force)[DIM];              // total force acting on the particle
static int (*NeighborFluidCount);             // [ParticleCount]
static int (*NeighborCount);                  // [ParticleCount]
static int (*Neighbor)[MAX_NEIGHBOR_COUNT];   // [ParticleCount]
static int (*NeighborCountP);                 // [ParticleCount]
static int (*NeighborP)[MAX_NEIGHBOR_COUNT];  // [ParticleCount]
static int    (*TmpIntScalar);                // [ParticleCount] to sort with cellId
static double (*TmpDoubleScalar);             // [ParticleCount]
static double (*TmpDoubleVector)[DIM];        // [ParticleCount]
#pragma acc declare create(ParticleCount,ParticleIndex,Property,Mass,Position,Velocity,Force)
#pragma acc declare create(NeighborFluidCount,NeighborCount,Neighbor,NeighborCountP,NeighborP)
#pragma acc declare create(TmpIntScalar,TmpDoubleScalar,TmpDoubleVector)


// BackGroundCells
#ifdef TWO_DIMENSIONAL
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1])
#else
#define CellId(iCX,iCY,iCZ)  (((iCX)%CellCount[0]+CellCount[0])%CellCount[0]*CellCount[1]*CellCount[2] + ((iCY)%CellCount[1]+CellCount[1])%CellCount[1]*CellCount[2] + ((iCZ)%CellCount[2]+CellCount[2])%CellCount[2])
#endif

static int PowerParticleCount;
static int ParticleCountPower;                   
static double CellWidth = 0.0;
static int CellCount[DIM];
static int TotalCellCount = 0;       // CellCount[0]*CellCount[1]*CellCount[2]
static int *CellFluidParticleBegin;  // [CellCounts+1]// beginning of fluid particles in the cell
static int *CellFluidParticleEnd;    // [CellCounts]  number of particles in the cell
static int *CellWallParticleBegin;   // [CellCounts]
static int *CellWallParticleEnd;     // [CellCounts]
static int *CellIndex;               // [ParticleCountPower>>1] fluid:CellId, wall:CellId+CellCounts, else:2*CellCounts 
static int *CellParticle;            // array of current particle id in the cells) [ParticleCountPower>>1]
#pragma acc declare create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount,TotalCellCount,CellFluidParticleBegin,CellFluidParticleEnd,CellWallParticleBegin,CellWallParticleEnd,CellIndex,CellParticle)

// Type
static double Density[TYPE_COUNT];
static double BulkModulus[TYPE_COUNT];
static double BulkViscosity[TYPE_COUNT];
static double ShearViscosity[TYPE_COUNT];
static double SurfaceTension[TYPE_COUNT];
static double CofA[TYPE_COUNT];   // coefficient for attractive pressure
static double CofK;               // coefficinet (ratio) for diffuse interface thickness normalized by ParticleSpacing
static double InteractionRatio[TYPE_COUNT][TYPE_COUNT];
#pragma acc declare create(Density,BulkModulus,BulkViscosity,ShearViscosity,SurfaceTension,CofA,CofK,InteractionRatio)


// Fluid
static int FluidParticleBegin;
static int FluidParticleEnd;
//static double KappaA;           // coefficient for attractive pressure
static double *DensityA;        // number density per unit volume for attractive pressure
static double (*GravityCenter)[DIM];
static double *PressureA;       // attractive pressure (surface tension)
static double *VolStrainP;        // number density per unit volume for base pressure
static double *DivergenceP;     // volumetric strainrate for pressure B
static double *PressureP;       // base pressure
static double *VirialPressureAtParticle; // VirialPressureInSingleParticleRegion
static double *VirialPressureInsideRadius; //VirialPressureInRegionInsideEffectiveRadius
static double (*VirialStressAtParticle)[DIM][DIM];
static double *Mu;              // viscosity coefficient for shear
static double *Lambda;          // viscosity coefficient for bulk
static double *Kappa;           // bulk modulus
#pragma acc declare create(FluidParticleBegin,FluidParticleEnd,DensityA,GravityCenter,PressureA,VolStrainP,DivergenceP,PressureP)
#pragma acc declare create(VirialPressureAtParticle,VirialPressureInsideRadius,VirialStressAtParticle,Mu,Lambda,Kappa)


static double Gravity[DIM] = {0.0,0.0,0.0};
#pragma acc declare create(Gravity)

// Wall
static int WallParticleBegin;
static int WallParticleEnd;
static double WallCenter[WALL_END][DIM];
static double WallVelocity[WALL_END][DIM];
static double WallOmega[WALL_END][DIM];
static double WallRotation[WALL_END][DIM][DIM];
#pragma acc declare create(WallParticleBegin,WallParticleEnd,WallCenter,WallVelocity,WallOmega,WallRotation)


// proceedures
static void readDataFile(char *filename);
static void readGridFile(char *filename);
static void writeProfFile(char *filename);
static void writeVtkFile(char *filename);
static void initializeWeight( void );
static void initializeDomain( void );
static void initializeFluid( void );
static void initializeWall( void );

static void calculateConvection();
static void calculateWall();
static void calculatePeriodicBoundary();
static void resetForce();
static void calculateCellParticle( void );
static void calculateNeighbor( void );
static void calculatePhysicalCoefficients( void );
static void calculateDensityA();
static void calculatePressureA();
static void calculateGravityCenter();
static void calculateDiffuseInterface();
static void calculateDensityP();
static void calculateDivergenceP();
static void calculatePressureP();
static void calculateViscosityV();
static void calculateGravity();
static void calculateAcceleration();
static void calculateMatrixA( void );
static void calculateMatrixC( void );
static void multiplyMatrixC( void );
static void solveWithConjugatedGradient( void );
static void calculateVirialPressureAtParticle();
static void calculateVirialPressureInsideRadius();
static void calculateVirialStressAtParticle();


// dual kernel functions
static double RadiusRatioA;
static double RadiusRatioG;
static double RadiusRatioP;
static double RadiusRatioV;

static double MaxRadius = 0.0;
static double RadiusA = 0.0;
static double RadiusG = 0.0;
static double RadiusP = 0.0;
static double RadiusV = 0.0;
static double Swa = 1.0;
static double Swg = 1.0;
static double Swp = 1.0;
static double Swv = 1.0;
static double N0a = 1.0;
static double N0p = 1.0;
static double R2g = 1.0;

#pragma acc declare create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)


#pragma acc routine seq
static double wa(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#else
    return 1.0/Swa * 1.0/(h*h*h) * (r/h)*(1.0-(r/h))*(1.0-(r/h));
#endif
}

#pragma acc routine seq
static double dwadr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swa * 1.0/(h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#else
    return 1.0/Swa * 1.0/(h*h*h) * (1.0-(r/h))*(1.0-3.0*(r/h))*(1.0/h);
#endif
}

#pragma acc routine seq
static double wg(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
static double dwgdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swg * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swg * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
inline double wp(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swp * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}


#pragma acc routine seq
inline double dwpdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swp * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swp * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}

#pragma acc routine seq
inline double wv(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * ((1.0-r/h)*(1.0-r/h));
#else    
    return 1.0/Swv * 1.0/(h*h*h) * ((1.0-r/h)*(1.0-r/h));
#endif
}

#pragma acc routine seq
inline double dwvdr(const double r, const double h){
#ifdef TWO_DIMENSIONAL
    return 1.0/Swv * 1.0/(h*h) * (-2.0/h*(1.0-r/h));
#else
    return 1.0/Swv * 1.0/(h*h*h) * (-2.0/h*(1.0-r/h));
#endif
}


	clock_t cFrom, cTill, cStart, cEnd;
	clock_t cNeigh=0, cExplicit=0, cImplicitTriplet=0, cImplicitInsert=0, cImplicitMatrix=0, cImplicitMulti=0, cImplicitSolve=0, cVirial=0, cOther=0;

int main(int argc, char *argv[])
{
	
    char logfilename[1024]  = DEFAULT_LOG;
    char datafilename[1024] = DEFAULT_DATA;
    char gridfilename[1024] = DEFAULT_GRID;
    char proffilename[1024] = DEFAULT_PROF;
    char vtkfilename[1024] = DEFAULT_VTK;
	    int numberofthread = 1;
    
    {
        if(argc>1)strcpy(datafilename,argv[1]);
        if(argc>2)strcpy(gridfilename,argv[2]);
        if(argc>3)strcpy(proffilename,argv[3]);
        if(argc>4)strcpy(vtkfilename,argv[4]);
        if(argc>5)strcpy( logfilename,argv[5]);
    	if(argc>6)numberofthread=atoi(argv[6]);
    }
    log_open(logfilename);
    {
        time_t t=time(NULL);
        log_printf("start reading files at %s\n",ctime(&t));
    }
	{
		#ifdef _OPENMP
		omp_set_num_threads( numberofthread );
		#pragma omp parallel
		{
			if(omp_get_thread_num()==0){
				log_printf("omp_get_num_threads()=%d\n", omp_get_num_threads() );
			}
		}
		#endif
    }
    readDataFile(datafilename);
    readGridFile(gridfilename);
    {
        time_t t=time(NULL);
        log_printf("start initialization at %s\n",ctime(&t));
    }
    initializeWeight();
    initializeFluid();
    initializeWall();
    initializeDomain();
	
	#pragma acc enter data create(ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc enter data create(ParticleCount,ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount])
	#pragma acc enter data create(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc enter data create(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc enter data create(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
	#pragma acc enter data create(NeighborFluidCount[0:ParticleCount],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
	#pragma acc enter data create(NeighborCountP[0:ParticleCount],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
	#pragma acc enter data create(TmpIntScalar[0:ParticleCount],TmpDoubleScalar[0:ParticleCount],TmpDoubleVector[0:ParticleCount][0:DIM])
	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],TotalCellCount)
	#pragma acc enter data create(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc enter data create(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc enter data create(FluidParticleBegin,FluidParticleEnd)
	#pragma acc enter data create(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
	#pragma acc enter data create(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
	#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount],VirialPressureInsideRadius[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc enter data create(Lambda[0:ParticleCount],Kappa[0:ParticleCount],Mu[0:ParticleCount])
	#pragma acc enter data create(Gravity[0:DIM])
	#pragma acc enter data create(WallParticleBegin,WallParticleEnd)
	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	
	#pragma acc update device(ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCount,ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount])
	#pragma acc update device(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc update device(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc update device(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
	#pragma acc update device(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],TotalCellCount)
	#pragma acc update device(Lambda[0:ParticleCount],Kappa[0:ParticleCount],Mu[0:ParticleCount])
	#pragma acc update device(Gravity[0:DIM])
	#pragma acc update device(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc update device(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)

	calculateCellParticle();
    calculateNeighbor();
    calculateDensityA();
	calculateGravityCenter();
	calculateDensityP();
    writeVtkFile("output.vtk");

	{
        time_t t=time(NULL);
        log_printf("start main roop at %s\n",ctime(&t));
    }
    int iStep=(int)(Time/Dt);
	cStart = clock();
	cFrom = cStart;
    while(Time < EndTime + 1.0e-5*Dt){
    	
    	if( Time + 1.0e-5*Dt >= OutputNext ){
            char filename[256];
            sprintf(filename,proffilename,iStep);
            writeProfFile(filename);
            log_printf("@ Prof Output Time : %e\n", Time );
            OutputNext += OutputInterval;
			cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;
        }
		
        // particle movement
        calculateConvection();

        // wall calculation
        calculateWall();

        // periodic boundary calculation
        calculatePeriodicBoundary();

        // reset Force to calculate conservative interaction
        resetForce();
    	cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;

        // calculate Neighbor
		calculateCellParticle();
        calculateNeighbor();
        cTill = clock(); cNeigh += (cTill-cFrom); cFrom = cTill;

        // calculate density
        calculateDensityA();
    	calculateGravityCenter();
    	calculateDensityP();

    	// calculate physical coefficient (viscosity, bulk modulus, bulk viscosity..)
		calculatePhysicalCoefficients();

        // calculate P(s,rho) s:fixed
        calculatePressureA();
    	
    	// calculate diffuse interface force
		calculateDiffuseInterface();
        
        // calculate Gravity
        calculateGravity();

        // calculate intermidiate Velocity
        calculateAcceleration();
        cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;
    	
       // calculate viscosity pressure implicitly
    	calculateMatrixA();
    	calculateMatrixC();
    	cTill = clock(); cImplicitMatrix += (cTill-cFrom); cFrom = cTill;
    	multiplyMatrixC();
    	cTill = clock(); cImplicitMulti += (cTill-cFrom); cFrom = cTill;
    	solveWithConjugatedGradient();
        cTill = clock(); cImplicitSolve += (cTill-cFrom); cFrom = cTill;    	
    	

        if( Time + 1.0e-5*Dt >= VtkOutputNext ){
 		   	calculateDivergenceP();
    		calculatePressureP();
        	calculateViscosityV();
    		calculateVirialStressAtParticle();
        	cTill = clock(); cVirial += (cTill-cFrom); cFrom = cTill;
            char filename [256];
            sprintf(filename,vtkfilename,iStep);
            writeVtkFile(filename);
            log_printf("@ Vtk Output Time : %e\n", Time );
            VtkOutputNext += VtkOutputInterval;
			cTill = clock(); cOther += (cTill-cFrom); cFrom = cTill;

        }

        Time += Dt;
        iStep++;
    	cTill = clock(); cExplicit += (cTill-cFrom); cFrom = cTill;

    }
	cEnd = cTill;
	
    {
        time_t t=time(NULL);
        log_printf("end main roop at %s\n",ctime(&t));
    	log_printf("neighbor search:         %lf [CPU sec]\n", (double)cNeigh/CLOCKS_PER_SEC);
    	log_printf("explicit calculation:    %lf [CPU sec]\n", (double)cExplicit/CLOCKS_PER_SEC);
    	log_printf("implicit triplet:        %lf [CPU sec]\n", (double)cImplicitTriplet/CLOCKS_PER_SEC);
    	log_printf("implicit insert:         %lf [CPU sec]\n", (double)cImplicitInsert/CLOCKS_PER_SEC);
    	log_printf("implicit matrix:         %lf [CPU sec]\n", (double)cImplicitMatrix/CLOCKS_PER_SEC);
    	log_printf("implicit multiplication: %lf [CPU sec]\n", (double)cImplicitMulti/CLOCKS_PER_SEC);
    	log_printf("implicit solve         : %lf [CPU sec]\n", (double)cImplicitSolve/CLOCKS_PER_SEC);
    	log_printf("virial calculation:      %lf [CPU sec]\n", (double)cVirial/CLOCKS_PER_SEC);
    	log_printf("other calculation:       %lf [CPU sec]\n", (double)cOther/CLOCKS_PER_SEC);
    	log_printf("total:                   %lf [CPU sec]\n", (double)(cNeigh+cExplicit+cImplicitTriplet+cImplicitInsert+cImplicitMatrix+cImplicitMulti+cImplicitSolve+cVirial+cOther)/CLOCKS_PER_SEC);
    	log_printf("total (check):           %lf [CPU sec]\n", (double)(cEnd-cStart)/CLOCKS_PER_SEC);
    }
	
	#pragma acc exit data delete(ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc exit data delete(ParticleCount,ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount])
	#pragma acc exit data delete(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
	#pragma acc exit data delete(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
	#pragma acc exit data delete(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(NeighborFluidCount[0:ParticleCount],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
	#pragma acc exit data delete(NeighborCountP[0:ParticleCount],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
	#pragma acc exit data delete(TmpIntScalar[0:ParticleCount],TmpDoubleScalar[0:ParticleCount],TmpDoubleVector[0:ParticleCount][0:DIM])
	#pragma acc exit data delete(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],TotalCellCount)
	#pragma acc exit data delete(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc exit data delete(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	#pragma acc exit data delete(FluidParticleBegin,FluidParticleEnd)
	#pragma acc exit data delete(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
	#pragma acc exit data delete(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
	#pragma acc exit data delete(VirialPressureAtParticle[0:ParticleCount],VirialPressureInsideRadius[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
	#pragma acc exit data delete(Lambda[0:ParticleCount],Kappa[0:ParticleCount],Mu[0:ParticleCount])
	#pragma acc exit data delete(Gravity[0:DIM])
	#pragma acc exit data delete(WallParticleBegin,WallParticleEnd)
	#pragma acc exit data delete(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
	#pragma acc exit data delete(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)


    return 0;

}

static void readDataFile(char *filename)
{
    FILE * fp;
    char buf[1024];
    const int reading_global=0;
    int mode=reading_global;
    

    fp=fopen(filename,"r");
    mode=reading_global;
    while(fp!=NULL && !feof(fp) && !ferror(fp)){
        if(fgets(buf,sizeof(buf),fp)!=NULL){
            if(buf[0]=='#'){}
            else if(sscanf(buf," Dt %lf",&Dt)==1){mode=reading_global;}
            else if(sscanf(buf," OutputInterval %lf",&OutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," VtkOutputInterval %lf",&VtkOutputInterval)==1){mode=reading_global;}
            else if(sscanf(buf," EndTime %lf",&EndTime)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioA %lf",&RadiusRatioA)==1){mode=reading_global;}
        	// else if(sscanf(buf," RadiusRatioG %lf",&RadiusRatioG)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioP %lf",&RadiusRatioP)==1){mode=reading_global;}
            else if(sscanf(buf," RadiusRatioV %lf",&RadiusRatioV)==1){mode=reading_global;}
            else if(sscanf(buf," Density %lf %lf %lf %lf",&Density[0],&Density[1],&Density[2],&Density[3])==4){mode=reading_global;}
            else if(sscanf(buf," BulkModulus %lf %lf %lf %lf",&BulkModulus[0],&BulkModulus[1],&BulkModulus[2],&BulkModulus[3])==4){mode=reading_global;}
            else if(sscanf(buf," BulkViscosity %lf %lf %lf %lf",&BulkViscosity[0],&BulkViscosity[1],&BulkViscosity[2],&BulkViscosity[3])==4){mode=reading_global;}
            else if(sscanf(buf," ShearViscosity %lf %lf %lf %lf",&ShearViscosity[0],&ShearViscosity[1],&ShearViscosity[2],&ShearViscosity[3])==4){mode=reading_global;}
            else if(sscanf(buf," SurfaceTension %lf %lf %lf %lf",&SurfaceTension[0],&SurfaceTension[1],&SurfaceTension[2],&SurfaceTension[3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type0) %lf %lf %lf %lf",&InteractionRatio[0][0],&InteractionRatio[0][1],&InteractionRatio[0][2],&InteractionRatio[0][3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type1) %lf %lf %lf %lf",&InteractionRatio[1][0],&InteractionRatio[1][1],&InteractionRatio[1][2],&InteractionRatio[1][3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type2) %lf %lf %lf %lf",&InteractionRatio[2][0],&InteractionRatio[2][1],&InteractionRatio[2][2],&InteractionRatio[2][3])==4){mode=reading_global;}
            else if(sscanf(buf," InteractionRatio(Type3) %lf %lf %lf %lf",&InteractionRatio[3][0],&InteractionRatio[3][1],&InteractionRatio[3][2],&InteractionRatio[3][3])==4){mode=reading_global;}
            else if(sscanf(buf," Gravity %lf %lf %lf", &Gravity[0], &Gravity[1], &Gravity[2])==3){mode=reading_global;}
            else if(sscanf(buf," Wall2  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[2][0],  &WallCenter[2][1],  &WallCenter[2][2],  &WallVelocity[2][0],  &WallVelocity[2][1],  &WallVelocity[2][2],  &WallOmega[2][0],  &WallOmega[2][1],  &WallOmega[2][2])==9){mode=reading_global;}
            else if(sscanf(buf," Wall3  Center %lf %lf %lf Velocity %lf %lf %lf Omega %lf %lf %lf", &WallCenter[3][0],  &WallCenter[3][1],  &WallCenter[3][2],  &WallVelocity[3][0],  &WallVelocity[3][1],  &WallVelocity[3][2],  &WallOmega[3][0],  &WallOmega[3][1],  &WallOmega[3][2])==9){mode=reading_global;}
            else{
                log_printf("Invalid line in data file \"%s\"\n", buf);
            }
        }
    }
    fclose(fp);
    return;
}

static void readGridFile(char *filename)
{
    FILE *fp=fopen(filename,"r");
    char buf[1024];   

    try{
        if(fgets(buf,sizeof(buf),fp)==NULL)throw;
        sscanf(buf,"%lf",&Time);
        if(fgets(buf,sizeof(buf),fp)==NULL)throw;
        sscanf(buf,"%d  %lf  %lf %lf %lf  %lf %lf %lf",
               &ParticleCount,
               &ParticleSpacing,
               &DomainMin[0], &DomainMax[0],
               &DomainMin[1], &DomainMax[1],
               &DomainMin[2], &DomainMax[2]);
#ifdef TWO_DIMENSIONAL
        ParticleVolume = ParticleSpacing*ParticleSpacing;
#else
	ParticleVolume = ParticleSpacing*ParticleSpacing*ParticleSpacing;
#endif
		ParticleIndex = (int *)malloc(ParticleCount*sizeof(int));
        Property = (int *)malloc(ParticleCount*sizeof(int));
        Position = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        Velocity = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        DensityA = (double *)malloc(ParticleCount*sizeof(double));
    	GravityCenter = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        PressureA = (double *)malloc(ParticleCount*sizeof(double));
        VolStrainP = (double *)malloc(ParticleCount*sizeof(double));
    	DivergenceP = (double *)malloc(ParticleCount*sizeof(double));
        PressureP = (double *)malloc(ParticleCount*sizeof(double));
        VirialPressureAtParticle = (double *)malloc(ParticleCount*sizeof(double));
        VirialPressureInsideRadius = (double *)malloc(ParticleCount*sizeof(double));
    	VirialStressAtParticle = (double (*) [DIM][DIM])malloc(ParticleCount*sizeof(double [DIM][DIM]));
    	Mass = (double (*))malloc(ParticleCount*sizeof(double));
        Force = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));
        Mu = (double (*))malloc(ParticleCount*sizeof(double));
    	Lambda = (double (*))malloc(ParticleCount*sizeof(double));
    	Kappa = (double (*))malloc(ParticleCount*sizeof(double));

    	TmpIntScalar = (int *)malloc(ParticleCount*sizeof(int));
    	TmpDoubleScalar = (double *)malloc(ParticleCount*sizeof(double));
    	TmpDoubleVector = (double (*)[DIM])malloc(ParticleCount*sizeof(double [DIM]));

        NeighborFluidCount = (int *)malloc(ParticleCount*sizeof(int));
    	NeighborCount  = (int *)malloc(ParticleCount*sizeof(int));
    	Neighbor       = (int (*)[MAX_NEIGHBOR_COUNT])malloc(ParticleCount*sizeof(int [MAX_NEIGHBOR_COUNT]));
    	NeighborCountP = (int *)malloc(ParticleCount*sizeof(int));
    	NeighborP      = (int (*)[MAX_NEIGHBOR_COUNT])malloc(ParticleCount*sizeof(int [MAX_NEIGHBOR_COUNT]));

        double (*q)[DIM] = Position;
        double (*v)[DIM] = Velocity;

        for(int iP=0;iP<ParticleCount;++iP){
            if(fgets(buf,sizeof(buf),fp)==NULL)break;
            sscanf(buf,"%d  %lf %lf %lf  %lf %lf %lf",
                   &Property[iP],
                   &q[iP][0],&q[iP][1],&q[iP][2],
                   &v[iP][0],&v[iP][1],&v[iP][2]
                   );
        }
    }catch(...){};

    fclose(fp);
	
	// set begin & end
	FluidParticleBegin=0;FluidParticleEnd=0;WallParticleBegin=0;WallParticleEnd=0;
    for(int iP=0;iP<ParticleCount;++iP){
    	if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END && WALL_BEGIN<=Property[iP+1] && Property[iP+1]<WALL_END){
    		FluidParticleEnd=iP+1;
    		WallParticleBegin=iP+1;
    	}
    	if(FLUID_BEGIN<=Property[iP] && Property[iP]<FLUID_END && iP+1==ParticleCount){
    		FluidParticleEnd=iP+1;
    	}
    	if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END && iP+1==ParticleCount){
    		WallParticleEnd=iP+1;
    	}
    }
	
    return;
}

static void writeProfFile(char *filename)
{
    FILE *fp=fopen(filename,"w");

    fprintf(fp,"%e\n",Time);
    fprintf(fp,"%d %e %e %e %e %e %e %e\n",
            ParticleCount,
            ParticleSpacing,
            DomainMin[0], DomainMax[0],
            DomainMin[1], DomainMax[1],
            DomainMin[2], DomainMax[2]);

    const double (*q)[DIM] = Position;
    const double (*v)[DIM] = Velocity;

    for(int iP=0;iP<ParticleCount;++iP){
            fprintf(fp,"%d %e %e %e  %e %e %e\n",
                    Property[iP],
                    q[iP][0], q[iP][1], q[iP][2],
                    v[iP][0], v[iP][1], v[iP][2]
            );
    }
    fflush(fp);
    fclose(fp);
}

static void writeVtkFile(char *filename)
{
	
//	#pragma acc update host(ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount])
//	#pragma acc update host(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
//	#pragma acc update host(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
//	#pragma acc update host(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
//	#pragma acc update host(VirialPressureAtParticle[0:ParticleCount],VirialPressureInsideRadius[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
//	#pragma acc update host(Lambda[0:ParticleCount],Kappa[0:ParticleCount],Mu[0:ParticleCount])
//	#pragma acc update host(NeighborFluidCount[0:ParticleCount],NeighborCount[0:ParticleCount])
//	#pragma acc update host(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
	
	// update parameters to be output
	#pragma acc update host(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],VirialPressureAtParticle[0:ParticleCount])
	#pragma acc update host(NeighborCount[0:ParticleCount],Force[0:ParticleCount][0:DIM])
	
	const double (*q)[DIM] = Position;
	const double (*v)[DIM] = Velocity;
	
	FILE *fp=fopen(filename, "w");
	
	fprintf(fp, "# vtk DataFile Version 2.0\n");
	fprintf(fp, "Unstructured Grid Example\n");
	fprintf(fp, "ASCII\n");
	
	fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");
	fprintf(fp, "POINTS %d float\n", ParticleCount);
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "%e %e %e\n", (float)q[iP][0], (float)q[iP][1], (float)q[iP][2]);
	}
	fprintf(fp, "CELLS %d %d\n", ParticleCount, 2*ParticleCount);
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "1 %d ",iP);
	}
	fprintf(fp, "\n");
	fprintf(fp, "CELL_TYPES %d\n", ParticleCount);
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "1 ");
	}
	fprintf(fp, "\n");
	
	fprintf(fp, "\n");
	
	fprintf(fp, "POINT_DATA %d\n", ParticleCount);
	fprintf(fp, "SCALARS label float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "%d\n", Property[iP]);
	}
	fprintf(fp, "\n");
//	fprintf(fp, "SCALARS Mass float 1\n");
//	fprintf(fp, "LOOKUP_TABLE default\n");
//	for(int iP=0;iP<ParticleCount;++iP){
//		fprintf(fp, "%e\n",(float) Mass[iP]);
//	}
//	fprintf(fp, "\n");
//	fprintf(fp, "SCALARS DensityA float 1\n");
//	fprintf(fp, "LOOKUP_TABLE default\n");
//	for(int iP=0;iP<ParticleCount;++iP){
//		fprintf(fp, "%e\n", (float)DensityA[iP]);
//	}
//	fprintf(fp, "\n");
//	fprintf(fp, "SCALARS PressureA float 1\n");
//	fprintf(fp, "LOOKUP_TABLE default\n");
//	for(int iP=0;iP<ParticleCount;++iP){
//		fprintf(fp, "%e\n", (float)PressureA[iP]);
//	}
//	fprintf(fp, "\n");
//	fprintf(fp, "SCALARS VolStrainP float 1\n");
//	fprintf(fp, "LOOKUP_TABLE default\n");
//	for(int iP=0;iP<ParticleCount;++iP){
//		fprintf(fp, "%e\n", (float)VolStrainP[iP]);
//	}
//	fprintf(fp, "\n");
//	fprintf(fp, "SCALARS DivergenceP float 1\n");
//	fprintf(fp, "LOOKUP_TABLE default\n");
//	for(int iP=0;iP<ParticleCount;++iP){
//		fprintf(fp, "%e\n", (float)DivergenceP[iP]);
//	}
//	fprintf(fp, "\n");
//	fprintf(fp, "SCALARS PressureP float 1\n");
//	fprintf(fp, "LOOKUP_TABLE default\n");
//	for(int iP=0;iP<ParticleCount;++iP){
//		fprintf(fp, "%e\n", (float)PressureP[iP]);
//	}
//	fprintf(fp, "\n");
	fprintf(fp, "SCALARS VirialPressureAtParticle float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "%e\n", (float)VirialPressureAtParticle[iP]); // trivial operation is done for 
	}
	fprintf(fp, "\n");
//	fprintf(fp, "SCALARS VirialPressureInsideRadius float 1\n");
//	fprintf(fp, "LOOKUP_TABLE default\n");
//	for(int iP=0;iP<ParticleCount;++iP){
//		fprintf(fp, "%e\n", (float)VirialPressureInsideRadius[iP]); // trivial operation is done for 
//	}
//	for(int iD=0;iD<DIM-1;++iD){
//		for(int jD=0;jD<DIM-1;++jD){
//			fprintf(fp, "\n");    fprintf(fp, "SCALARS VirialStressAtParticle[%d][%d] float 1\n",iD,jD);
//			fprintf(fp, "LOOKUP_TABLE default\n");
//			for(int iP=0;iP<ParticleCount;++iP){
//				fprintf(fp, "%e\n", (float)VirialStressAtParticle[iP][iD][jD]); // trivial operation is done for 
//			}
//		}
//	}
//	fprintf(fp, "\n");
//	fprintf(fp, "SCALARS Mu float 1\n");
//	fprintf(fp, "LOOKUP_TABLE default\n");
//	for(int iP=0;iP<ParticleCount;++iP){
//		fprintf(fp, "%e\n", (float)Mu[iP]);
//	}
//	fprintf(fp, "\n");
	fprintf(fp, "SCALARS neighbor float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "%d\n", NeighborCount[iP]);
	}
	fprintf(fp, "\n");
	//	fprintf(fp, "SCALARS neighborFluid float 1\n");
	//	fprintf(fp, "LOOKUP_TABLE default\n");
	//	for(int iP=0;iP<ParticleCount;++iP){
		//		fprintf(fp, "%d\n", NeighborFluidCount[iP]);
		//	}
	//	fprintf(fp, "\n");
//	fprintf(fp, "SCALARS particleId float 1\n");
//	fprintf(fp, "LOOKUP_TABLE default\n");
	//	for(int iP=0;iP<ParticleCount;++iP){
		//		fprintf(fp, "%d\n", ParticleIndex[iP]);
		//	}
	//	fprintf(fp, "\n");
	//	fprintf(fp, "SCALARS cellId float 1\n");
	//	fprintf(fp, "LOOKUP_TABLE default\n");
	//	for(int iP=0;iP<ParticleCount;++iP){
		//		fprintf(fp, "%d\n", CellIndex[iP]);
		//	}
	//	fprintf(fp, "\n");
	//	fprintf(fp, "SCALARS cellPcl float 1\n");
	//	fprintf(fp, "LOOKUP_TABLE default\n");
	//	for(int iP=0;iP<ParticleCount;++iP){
		//		fprintf(fp, "%d\n", CellParticle[iP]);
		//	}
	//	fprintf(fp, "\n");
	fprintf(fp, "VECTORS velocity float\n");
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "%e %e %e\n", (float)v[iP][0], (float)v[iP][1], (float)v[iP][2]);
	}
	fprintf(fp, "\n");
//	fprintf(fp, "VECTORS GravityCenter float\n");
//	for(int iP=0;iP<ParticleCount;++iP){
//		fprintf(fp, "%e %e %e\n", (float)GravityCenter[iP][0], (float)GravityCenter[iP][1], (float)GravityCenter[iP][2]);
//	}
//	fprintf(fp, "\n");
	fprintf(fp, "VECTORS force float\n");
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "%e %e %e\n", (float)Force[iP][0], (float)Force[iP][1], (float)Force[iP][2]);
	}
	fprintf(fp, "\n");
	
	fflush(fp);
	fclose(fp);
}


static void initializeWeight()
{
	RadiusRatioG = RadiusRatioA;
	
	RadiusA = RadiusRatioA*ParticleSpacing;
	RadiusG = RadiusRatioG*ParticleSpacing;
	RadiusP = RadiusRatioP*ParticleSpacing;
	RadiusV = RadiusRatioV*ParticleSpacing;
	
	
#ifdef TWO_DIMENSIONAL
		Swa = 1.0/2.0 * 2.0/15.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swg = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swp = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		Swv = 1.0/2.0 * 1.0/3.0 * M_PI /ParticleSpacing/ParticleSpacing;
		R2g = 1.0/2.0 * 1.0/30.0* M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing /Swg;
#else
		//code for three dimensional
		Swa = 1.0/3.0 * 1.0/5.0*M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swg = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swp = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		Swv = 1.0/3.0 * 2.0/5.0 * M_PI /ParticleSpacing/ParticleSpacing/ParticleSpacing;
		R2g = 1.0/3.0 * 4.0/105.0*M_PI *RadiusG*RadiusG /ParticleSpacing/ParticleSpacing/ParticleSpacing /Swg;
#endif
	
	
	    {// N0a
        const double radius_ratio = RadiusA/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusA*RadiusA){
                        const double rij = sqrt(rij2);
                        const double wij = wa(rij,RadiusA);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusA*RadiusA){
                            const double rij = sqrt(rij2);
                            const double wij = wa(rij,RadiusA);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0a = sum;
        log_printf("N0a = %e, count=%d\n", N0a, count);
    }	

    {// N0p
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)(radius_ratio +3.0);
        int count = 0;
        double sum = 0.0;
#ifdef TWO_DIMENSIONAL
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                if(!(iX==0 && iY==0)){
                	const double x = ParticleSpacing * ((double)iX);
                	const double y = ParticleSpacing * ((double)iY);
                    const double rij2 = x*x + y*y;
                    if(rij2<=RadiusP*RadiusP){
                        const double rij = sqrt(rij2);
                        const double wij = wp(rij,RadiusP);
                        sum += wij;
                        count ++;
                    }
                }
            }
        }
#else
        for(int iX=-range;iX<=range;++iX){
            for(int iY=-range;iY<=range;++iY){
                for(int iZ=-range;iZ<=range;++iZ){
                    if(!(iX==0 && iY==0 && iZ==0)){
                    	const double x = ParticleSpacing * ((double)iX);
                    	const double y = ParticleSpacing * ((double)iY);
                    	const double z = ParticleSpacing * ((double)iZ);
                        const double rij2 = x*x + y*y + z*z;
                        if(rij2<=RadiusP*RadiusP){
                            const double rij = sqrt(rij2);
                            const double wij = wp(rij,RadiusP);
                            sum += wij;
                            count ++;
                        }
                    }
                }
            }
        }
#endif
        N0p = sum;
        log_printf("N0p = %e, count=%d\n", N0p, count);
    }				
}


static void initializeFluid()
{
	for(int iP=0;iP<ParticleCount;++iP){
		Mass[iP]=Density[Property[iP]]*ParticleVolume;
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Kappa[iP]=BulkModulus[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Lambda[iP]=BulkViscosity[Property[iP]];
	}
	for(int iP=0;iP<ParticleCount;++iP){
		Mu[iP]=ShearViscosity[Property[iP]];
	}
	#ifdef TWO_DIMENSIONAL
	CofK = 0.350778153;
	double integN=0.024679383;
	double integX=0.226126699;
	#else 
	CofK = 0.326976006;
	double integN=0.021425779;
	double integX=0.233977488;
	#endif
	
	for(int iT=0;iT<TYPE_COUNT;++iT){
		CofA[iT]=SurfaceTension[iT] / ((RadiusG/ParticleSpacing)*(integN+CofK*CofK*integX));
	}
}

static void initializeWall()
{
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		
		double theta;
		double normal[DIM]={0.0,0.0,0.0};
		double q[DIM+1];
		double t[DIM];
		double (&R)[DIM][DIM]=WallRotation[iProp];
		
		theta = abs(WallOmega[iProp][0]*WallOmega[iProp][0]+WallOmega[iProp][1]*WallOmega[iProp][1]+WallOmega[iProp][2]*WallOmega[iProp][2]);
		if(theta!=0.0){
			for(int iD=0;iD<DIM;++iD){
				normal[iD]=WallOmega[iProp][iD]/theta;
			}
		}
		q[0]=normal[0]*sin(theta*Dt/2.0);
		q[1]=normal[1]*sin(theta*Dt/2.0);
		q[2]=normal[2]*sin(theta*Dt/2.0);
		q[3]=cos(theta*Dt/2.0);
		t[0]=WallVelocity[iProp][0]*Dt;
		t[1]=WallVelocity[iProp][1]*Dt;
		t[2]=WallVelocity[iProp][2]*Dt;
		
		R[0][0] = q[0]*q[0]-q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[0][1] = 2.0*(q[0]*q[1]-q[2]*q[3]);
		R[0][2] = 2.0*(q[0]*q[2]+q[1]*q[3]);
		
		R[1][0] = 2.0*(q[0]*q[1]+q[2]*q[3]);
		R[1][1] = -q[0]*q[0]+q[1]*q[1]-q[2]*q[2]+q[3]*q[3];
		R[1][2] = 2.0*(q[1]*q[2]-q[0]*q[3]);
		
		R[2][0] = 2.0*(q[0]*q[2]-q[1]*q[3]);
		R[2][1] = 2.0*(q[1]*q[2]+q[0]*q[3]);
		R[2][2] = -q[0]*q[0]-q[1]*q[1]+q[2]*q[2]+q[3]*q[3];
	}
	
}

static void initializeDomain( void )
{
    CellWidth = ParticleSpacing;
    
    double cellCount[DIM];

    cellCount[0] = round((DomainMax[0] - DomainMin[0])/CellWidth);
    cellCount[1] = round((DomainMax[1] - DomainMin[1])/CellWidth);
#ifdef TWO_DIMENSIONAL
    cellCount[2] = 1;
#else
    cellCount[2] = round((DomainMax[2] - DomainMin[2])/CellWidth);
#endif

    CellCount[0] = (int)cellCount[0];
    CellCount[1] = (int)cellCount[1];
    CellCount[2] = (int)cellCount[2];
	TotalCellCount = CellCount[0]*CellCount[1]*CellCount[2];
	fprintf(stderr, "TotalCellCount = %d\n", TotalCellCount);

    if(cellCount[0]!=(double)CellCount[0] || cellCount[1]!=(double)CellCount[1] ||cellCount[2]!=(double)CellCount[2]){
        fprintf(stderr,"DomainWidth/CellWidth is not integer\n");
        DomainMax[0] = DomainMin[0] + CellWidth*(double)CellCount[0];
        DomainMax[1] = DomainMin[1] + CellWidth*(double)CellCount[1];
        DomainMax[2] = DomainMin[2] + CellWidth*(double)CellCount[2];
        fprintf(stderr,"Changing the Domain Max to (%e,%e,%e)\n", DomainMax[0], DomainMax[1], DomainMax[2]);
    }
    DomainWidth[0] = DomainMax[0] - DomainMin[0];
    DomainWidth[1] = DomainMax[1] - DomainMin[1];
    DomainWidth[2] = DomainMax[2] - DomainMin[2];

	CellFluidParticleBegin = (int *)malloc( TotalCellCount * sizeof(int) );
	CellFluidParticleEnd   = (int *)malloc( TotalCellCount * sizeof(int) );
	CellWallParticleBegin  = (int *)malloc( TotalCellCount * sizeof(int) );
	CellWallParticleEnd    = (int *)malloc( TotalCellCount * sizeof(int) );
	
	// calculate minimun PowerParticleCount which sataisfies  ParticleCount < PowerParticleCount = pow(2,ParticleCountPower) 
	ParticleCountPower=0;
	while((ParticleCount>>ParticleCountPower)!=0){
		++ParticleCountPower;
	}
	PowerParticleCount = (1<<ParticleCountPower);
	fprintf(stderr,"memory for CellIndex and CellParticle %d\n", PowerParticleCount );
	CellIndex    = (int *)malloc( (PowerParticleCount) * sizeof(int) );
    CellParticle = (int *)malloc( (PowerParticleCount) * sizeof(int) );
	
	MaxRadius = ((RadiusA>MaxRadius) ? RadiusA : MaxRadius);
	MaxRadius = ((2.0*RadiusP>MaxRadius) ? 2.0*RadiusP : MaxRadius);
	MaxRadius = ((RadiusV>MaxRadius) ? RadiusV : MaxRadius);
	fprintf(stderr, "MaxRadius = %lf\n", MaxRadius);
}

static void calculateCellParticle()
{
	// store and sort with cells
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0; iP<PowerParticleCount; ++iP){
		if(iP<ParticleCount){
			const int iCX=((int)floor((Position[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
			const int iCY=((int)floor((Position[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
			const int iCZ=((int)floor((Position[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];
			CellIndex[iP]=CellId(iCX,iCY,iCZ);
			if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END){
				CellIndex[iP] += TotalCellCount;
			}
			CellParticle[iP]=iP;
		}
		else{
			CellIndex[ iP ]    = 2*TotalCellCount;
			CellParticle[ iP ] = ParticleCount;
		}
	}
	
	// sort with CellIndex
	// https://edom18.hateblo.jp/entry/2020/09/21/150416
	for(int iMain=0;iMain<ParticleCountPower;++iMain){
		for(int iSub=0;iSub<=iMain;++iSub){
			
			int dist = (1<< (iMain-iSub));
			
			#pragma acc kernels present(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
			#pragma acc loop independent
			#pragma omp parallel for
			for(int iP=0;iP<(1<<ParticleCountPower);++iP){
				bool up = ((iP >> iMain) & 2) == 0;
				
				if(  (( iP & dist )==0) && ( CellIndex[ iP ] > CellIndex[ iP | dist ] == up) ){
					int tmpCellIndex    = CellIndex[ iP ];
					int tmpCellParticle = CellParticle[ iP ];
					CellIndex[ iP ]     = CellIndex[ iP | dist ];
					CellParticle[ iP ]  = CellParticle[ iP | dist ];
					CellIndex[ iP | dist ]    = tmpCellIndex;
					CellParticle[ iP | dist ] = tmpCellParticle;
				}
			}
		}
	}
	
	// search for CellFluidParticleBegin[iC]
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		CellFluidParticleBegin[iC]=0;
		CellFluidParticleEnd[iC]=0;
		CellWallParticleBegin[iC]=0;
		CellWallParticleEnd[iC]=0;
	}
	
	int threshold[4]={0,0,0,0};
	
	#pragma acc kernels create(threshold[0:4])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iDummy=0;iDummy<4;++iDummy){
		threshold[iDummy]=0;
	}
	
	#pragma acc kernels copyout(threshold[0:4]) present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=1; iP<ParticleCount+1; ++iP){
		if( CellIndex[iP-1]<CellIndex[iP] ){
			if( CellIndex[iP-1] < TotalCellCount ){
				CellFluidParticleEnd[ CellIndex[iP-1] ] = iP;
			}
			else if( CellIndex[iP-1] - TotalCellCount < TotalCellCount ){
				CellWallParticleEnd[ CellIndex[iP-1]-TotalCellCount ] = iP;
			}
			if( CellIndex[iP] < TotalCellCount ){
				CellFluidParticleBegin[ CellIndex[iP] ] = iP;
			}
			else if( CellIndex[iP] - TotalCellCount < TotalCellCount ){
				CellWallParticleBegin[ CellIndex[iP]-TotalCellCount ] = iP;
			}
			if( CellIndex[iP-1]/TotalCellCount < CellIndex[iP]/TotalCellCount ){
				if( CellIndex[iP-1] < TotalCellCount ){
					threshold[1] = iP;
				}
				else if( CellIndex[iP-1] - TotalCellCount < TotalCellCount ){
					threshold[3] = iP;
				}
				if( CellIndex[iP] < TotalCellCount ){
					threshold[0] = iP;
				}
				else if( CellIndex[iP] - TotalCellCount < TotalCellCount ){
					threshold[2] = iP;
				}
			}
		}
	}
	
	FluidParticleBegin = threshold[0];
	FluidParticleEnd   = threshold[1];
	WallParticleBegin  = threshold[2];
	WallParticleEnd    = threshold[3];
	
	
	// re-arange particles in CellIndex order
	#pragma acc kernels present(ParticleIndex[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		TmpIntScalar[iP]=ParticleIndex[CellParticle[iP]];
	}
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		ParticleIndex[iP]=TmpIntScalar[iP];
	}
	
	#pragma acc kernels present(Property[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		TmpIntScalar[iP]=Property[CellParticle[iP]];
	}
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Property[iP]=TmpIntScalar[iP];
	}
	
	#pragma acc kernels present(Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			TmpDoubleVector[iP][iD]=Position[CellParticle[iP]][iD];
		}
	}
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Position[iP][iD]=TmpDoubleVector[iP][iD];
		}
	}
		
	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			TmpDoubleVector[iP][iD]=Velocity[CellParticle[iP]][iD];
		}
	}
	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Velocity[iP][iD]=TmpDoubleVector[iP][iD];
		}
	}
}



static void calculateNeighbor( void )
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		NeighborFluidCount[iP]=0;
		NeighborCount[iP]=0;
		for(int iN=0;iN<MAX_NEIGHBOR_COUNT;++iN){
			Neighbor[iP][iN]=-1;
		}
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		NeighborCountP[iP]=0;
		for(int iN=0;iN<MAX_NEIGHBOR_COUNT;++iN){
			NeighborP[iP][iN]=-1;
		}
	}
	
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		const int range = (int)(ceil(MaxRadius/CellWidth));
		
		const int iCX=(CellIndex[iP]/(CellCount[1]*CellCount[2]))%TotalCellCount;
		const int iCY=(CellIndex[iP]/CellCount[2])%CellCount[1];
		const int iCZ=CellIndex[iP]%CellCount[2];
		// same as
		// const int iCX=((int)floor((q[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
		// const int iCY=((int)floor((q[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
		// const int iCZ=((int)floor((q[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];
		
		#ifdef TWO_DIMENSIONAL
		#pragma acc loop seq
		for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
			#pragma acc loop seq
			for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
				const int jCZ=0;
				const int jC=CellId(jCX,jCY,jCZ);
				#pragma acc loop seq
				for(int jP=CellFluidParticleBegin[jC];jP<CellFluidParticleEnd[jC];++jP){
					double qij[DIM];
					#pragma acc loop seq
					for(int iD=0;iD<DIM;++iD){
						qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
					}
					const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
					if(qij2 <= MaxRadius*MaxRadius){
						if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
							NeighborCount[iP]++;
							NeighborFluidCount[iP]++;
							continue;
						}
						Neighbor[iP][NeighborCount[iP]]=jP;
						NeighborCount[iP]++;
						NeighborFluidCount[iP]++;
						
						if(qij2 <= RadiusP*RadiusP){
							NeighborP[iP][NeighborCountP[iP]]=jP;
							NeighborCountP[iP]++;
						}
					}
				}
			}
		}
		#pragma acc loop seq
		for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
			#pragma acc loop seq
			for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
				const int jCZ=0;
				const int jC=CellId(jCX,jCY,jCZ);
				#pragma acc loop seq
				for(int jP=CellWallParticleBegin[jC];jP<CellWallParticleEnd[jC];++jP){
					double qij[DIM];
					#pragma acc loop seq
					for(int iD=0;iD<DIM;++iD){
						qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
					}
					const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
					if(qij2 <= MaxRadius*MaxRadius){
						if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
							NeighborCount[iP]++;
							continue;
						}
						Neighbor[iP][NeighborCount[iP]]=jP;
						NeighborCount[iP]++;
						
						if(qij2 <= RadiusP*RadiusP){
							NeighborP[iP][NeighborCountP[iP]]=jP;
							NeighborCountP[iP]++;
						}
					}
				}
			}
		}
		
		#else // TWO_DIMENSIONAL
		#pragma acc loop seq
		for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
			#pragma acc loop seq
			for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
				#pragma acc loop seq
				for(int jCZ=iCZ-range;jCZ<=iCZ+range;++jCZ){
					const int jC=CellId(jCX,jCY,jCZ);
					#pragma acc loop seq
					for(int jP=CellFluidParticleBegin[jC];jP<CellFluidParticleEnd[jC];++jP){
						double qij[DIM];
						#pragma acc loop seq
						for(int iD=0;iD<DIM;++iD){
							qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
						}
						const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
						if(qij2 <= MaxRadius*MaxRadius){
							if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
								NeighborCount[iP]++;
								NeighborFluidCount[iP]++;
								continue;
							}
							Neighbor[iP][NeighborCount[iP]]=jP;
							NeighborCount[iP]++;
							NeighborFluidCount[iP]++;
						
							if(qij2 <= RadiusP*RadiusP){
								NeighborP[iP][NeighborCountP[iP]]=jP;
								NeighborCountP[iP]++;
							}
						}
					}
				}
			}
		}
		#pragma acc loop seq
		for(int jCX=iCX-range;jCX<=iCX+range;++jCX){
			#pragma acc loop seq
			for(int jCY=iCY-range;jCY<=iCY+range;++jCY){
				#pragma acc loop seq
				for(int jCZ=iCZ-range;jCZ<=iCZ+range;++jCZ){
					const int jC=CellId(jCX,jCY,jCZ);
					#pragma acc loop seq
					for(int jP=CellWallParticleBegin[jC];jP<CellWallParticleEnd[jC];++jP){
						double qij[DIM];
						#pragma acc loop seq
						for(int iD=0;iD<DIM;++iD){
							qij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
						}
						const double qij2= qij[0]*qij[0]+qij[1]*qij[1]+qij[2]*qij[2];
						if(qij2 <= MaxRadius*MaxRadius){
							if(NeighborCount[iP]>=MAX_NEIGHBOR_COUNT){
								NeighborCount[iP]++;
								continue;
							}
							Neighbor[iP][NeighborCount[iP]]=jP;
							NeighborCount[iP]++;
							
							if(qij2 <= RadiusP*RadiusP){
								NeighborP[iP][NeighborCountP[iP]]=jP;
								NeighborCountP[iP]++;
							}
						}
					}
				}
			}
		}
		#endif // TWO_DIMENSIONAL
	}
}
 
static void calculateConvection()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
        Position[iP][0] += Velocity[iP][0]*Dt;
        Position[iP][1] += Velocity[iP][1]*Dt;
        Position[iP][2] += Velocity[iP][2]*Dt;
    }
}

static void resetForce()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	#pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Force[iP][iD]=0.0;
        }
    }
}


static void calculatePhysicalCoefficients()
{	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Mass[iP]=Density[Property[iP]]*ParticleVolume;
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Kappa[iP]=BulkModulus[Property[iP]];
		if(VolStrainP[iP]<0.0){Kappa[iP]=0.0;}
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Lambda[iP]=BulkViscosity[Property[iP]];
		if(VolStrainP[iP]<0.0){Lambda[iP]=0.0;}
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		Mu[iP]=ShearViscosity[Property[iP]];
	}
}

static void calculateDensityA()
{
    
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM])
	{	
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<ParticleCount;++iP){
			double sum = 0.0;
			#pragma acc loop seq
			for(int iN=0;iN<NeighborCount[iP];++iN){
				const int jP=Neighbor[iP][iN];
				if(iP==jP)continue;
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				double xij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
				}
				const double radius = RadiusA;
				const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
				if(radius*radius - rij2 >= 0){
					const double rij = sqrt(rij2);
					const double weight = ratio * wa(rij,radius);
					sum += weight;
				}
			}
			DensityA[iP]=sum;
		}
	}
}

static void calculateGravityCenter()
{
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM])
	
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iP=0;iP<ParticleCount;++iP){
			double sum[DIM]={0.0,0.0,0.0};
			#pragma acc loop seq
			for(int iN=0;iN<NeighborCount[iP];++iN){
				const int jP=Neighbor[iP][iN];
				if(iP==jP)continue;
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				double xij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
				}
				const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
				if(RadiusG*RadiusG - rij2 >= 0){
					const double rij = sqrt(rij2);
					const double weight = ratio * wg(rij,RadiusG);
					#pragma acc loop seq
					for(int iD=0;iD<DIM;++iD){
						sum[iD] += xij[iD]*weight/R2g*RadiusG;
					}
				}
			}
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				GravityCenter[iP][iD] = sum[iD];
			}
		}
	
}

static void calculatePressureA()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureA[iP] = CofA[Property[iP]]*(DensityA[iP]-N0a)/ParticleSpacing;
		if(N0a<=DensityA[iP]){
			PressureA[iP] = 0.0;
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	double force[DIM]={0.0,0.0,0.0};
    	#pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
        	if(iP==jP)continue;
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
        	double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double radius = RadiusA;
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(radius*radius - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = ratio_ij * dwadr(rij,radius);
            	const double dwji = ratio_ji * dwadr(rij,radius);
                const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
            	#pragma acc loop seq
                for(int iD=0;iD<DIM;++iD){
                    force[iD] += (PressureA[iP]*dwij+PressureA[jP]*dwji)*eij[iD]* ParticleVolume;
                }
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}

static void calculateDiffuseInterface()
{
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],GravityCenter[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		const double ai = CofA[Property[iP]]*(CofK)*(CofK);
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			if(iP==jP)continue;
			const double aj = CofA[Property[iP]]*(CofK)*(CofK);
			double ratio_ij = InteractionRatio[Property[iP]][Property[jP]];
			double ratio_ji = InteractionRatio[Property[jP]][Property[iP]];
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(RadiusG*RadiusG - rij2 > 0){
				const double rij = sqrt(rij2);
				const double wij = ratio_ij * wg(rij,RadiusG);
				const double wji = ratio_ji * wg(rij,RadiusG);
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (aj*GravityCenter[jP][iD]*wji-ai*GravityCenter[iP][iD]*wij)/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				const double dwij = ratio_ij * dwgdr(rij,RadiusG);
				const double dwji = ratio_ji * dwgdr(rij,RadiusG);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (aj*GravityCenter[jP][iD]*dwji-ai*GravityCenter[iP][iD]*dwij)*xij[iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] -= (gr)*eij[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}

static void calculateDensityP()
{
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCountP[iP];++iN){
			const int jP=NeighborP[iP][iN];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double weight = wp(rij,radius);
				sum += weight;
			}
		}
		VolStrainP[iP] = (sum - N0p);
	}
}

static void calculateDivergenceP()
{

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double sum = 0.0;
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCountP[iP];++iN){
            const int jP=NeighborP[iP][iN];
			if(iP==jP)continue;
            double xij[DIM];
			#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 >= 0){
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					DivergenceP[iP] -= uij[iD]*eij[iD]*dw;
				}
			}
		}
		DivergenceP[iP]=sum;
	}
}

static void calculatePressureP()
{
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		PressureP[iP] = -Lambda[iP]*DivergenceP[iP];
		if(VolStrainP[iP]>0.0){
			PressureP[iP]+=Kappa[iP]*VolStrainP[iP];
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],PressureP[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double force[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCountP[iP];++iN){
            const int jP=NeighborP[iP][iN];
			if(iP==jP)continue;
            double xij[DIM];
			#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
			const double radius = RadiusP;
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			if(radius*radius - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dw = dwpdr(rij,radius);
				double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					force[iD] += (PressureP[iP]+PressureP[jP])*gradw[iD]*ParticleVolume;
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			Force[iP][iD]+=force[iD];
		}
	}
}

static void calculateViscosityV(){

	#pragma acc kernels present(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Mu[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	double force[DIM]={0.0,0.0,0.0};
    	#pragma acc loop seq
        for(int iN=0;iN<NeighborCount[iP];++iN){
            const int jP=Neighbor[iP][iN];
        	if(iP==jP)continue;
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(Position[jP][iD] - Position[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            
        	// viscosity term
        	if(RadiusV*RadiusV - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = -dwvdr(rij,RadiusV);
            	const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
        		double uij[DIM];
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					uij[iD]=Velocity[jP][iD]-Velocity[iP][iD];
				}
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
            	double fij[DIM] = {0.0,0.0,0.0};
        		#pragma acc loop seq
            	for(int iD=0;iD<DIM;++iD){
            		#ifdef TWO_DIMENSIONAL
            		force[iD] += 8.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#else
            		force[iD] += 10.0*muij*(uij[0]*eij[0]+uij[1]*eij[1]+uij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
            		#endif
            	}
            }
        }
    	#pragma acc loop seq
    	for(int iD=0;iD<DIM;++iD){
    		Force[iP][iD] += force[iD];
    	}
    }
}

static void calculateGravity(){
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
		Force[iP][0] += Mass[iP]*Gravity[0];
		Force[iP][1] += Mass[iP]*Gravity[1];
		Force[iP][2] += Mass[iP]*Gravity[2];
	}
}

static void calculateAcceleration()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
		Velocity[iP][0] += Force[iP][0]/Mass[iP]*Dt;
		Velocity[iP][1] += Force[iP][1]/Mass[iP]*Dt;
		Velocity[iP][2] += Force[iP][2]/Mass[iP]*Dt;
	}
}


static void calculateWall()
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
        Force[iP][0] = 0.0;
        Force[iP][1] = 0.0;
        Force[iP][2] = 0.0;
    }
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=WallParticleBegin;iP<WallParticleEnd;++iP){
		const int iProp = Property[iP];
		double r[DIM] = {Position[iP][0]-WallCenter[iProp][0],Position[iP][1]-WallCenter[iProp][1],Position[iP][2]-WallCenter[iProp][2]};
		const double (&R)[DIM][DIM] = WallRotation[iProp];
		const double (&w)[DIM] = WallOmega[iProp];
		r[0] = R[0][0]*r[0]+R[0][1]*r[1]+R[0][2]*r[2];
		r[1] = R[1][0]*r[0]+R[1][1]*r[1]+R[1][2]*r[2];
		r[2] = R[2][0]*r[0]+R[2][1]*r[1]+R[2][2]*r[2];
		Velocity[iP][0] = w[1]*r[2]-w[2]*r[1] + WallVelocity[iProp][0];
		Velocity[iP][1] = w[2]*r[0]-w[0]*r[2] + WallVelocity[iProp][1];
		Velocity[iP][2] = w[0]*r[1]-w[1]*r[0] + WallVelocity[iProp][2];
		Position[iP][0] = r[0] + WallCenter[iProp][0] + WallVelocity[iProp][0]*Dt;
		Position[iP][1] = r[1] + WallCenter[iProp][1] + WallVelocity[iProp][1]*Dt;
		Position[iP][2] = r[2] + WallCenter[iProp][2] + WallVelocity[iProp][2]*Dt;
		
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iProp=WALL_BEGIN;iProp<WALL_END;++iProp){
		WallCenter[iProp][0] += WallVelocity[iProp][0]*Dt;
		WallCenter[iProp][1] += WallVelocity[iProp][1]*Dt;
		WallCenter[iProp][2] += WallVelocity[iProp][2]*Dt;
	}
}

static int NonzeroCountA;
static double *CsrCofA;  // [ FluidCount * DIM x NeighFluidCount * DIM]
static int    *CsrIndA;  // [ FluidCount * DIM x NeighFluidCount * DIM]
static int    *CsrPtrA;  // [ FluidCount * DIM + 1 ] NeighborFluidCount‚ÌÏŽZ—ñ‚Æ‚µ‚ÄŒvŽZ‰Â
static double *VectorB;  // [ FluidCount * DIM ]
#pragma acc declare create(NonzeroCountA,CsrCofA,CsrIndA,CsrPtrA,VectorB)

static void calculateMatrixA( void )
{
	const double (*r)[DIM] = Position;
    const double (*v)[DIM] = Velocity;
	const double (*m) = Mass;
	
    
	// Copy DIM*NeighborFluidCount to CsrPtrA
	int power = 0;
	const int fluidcount = FluidParticleEnd-FluidParticleBegin;
	const int N = DIM*(fluidcount);
	while( (N>>power) != 0 ){
		power++;
	}
	const int powerN = (1<<power);
	
	CsrPtrA = (int *)malloc( powerN * sizeof(int));
	#pragma acc enter data create(CsrPtrA[0:powerN])
	
    #pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<powerN;++iRow){
		CsrPtrA[iRow]=0;
	}

    #pragma acc kernels present(CsrPtrA[0:powerN])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<fluidcount;++iP){
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow = DIM*iP+rD+1;
			CsrPtrA[iRow] = DIM*NeighborFluidCount[iP];
		}
	}
	
	// Convert CsrPtrA into cumulative sum
    for(int iMain=0;iMain<power;++iMain){
		const int dist = (1<<iMain);	
		#pragma acc kernels present(CsrPtrA[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			CsrPtrA[iRow]+=CsrPtrA[iRow+dist];
		}
	}
    for(int iMain=0;iMain<power;++iMain){
		const int dist = (powerN>>(iMain+1));	
		#pragma acc kernels present(CsrPtrA[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			CsrPtrA[iRow]-=CsrPtrA[iRow+dist];
			CsrPtrA[iRow+dist]+=CsrPtrA[iRow];
		}
	}
	
    #pragma acc kernels present(CsrPtrA[0:powerN])
	#pragma acc loop seq
	for(int iDummy=0;iDummy<1;++iDummy){
		NonzeroCountA=CsrPtrA[N];
	}
	#pragma acc update host(NonzeroCountA)
	
	// calculate coeeficient matrix A and source vector B
	CsrCofA = (double *)malloc( NonzeroCountA * sizeof(double) );
	CsrIndA = (int *)malloc( NonzeroCountA * sizeof(int) );
	VectorB = (double *)malloc( N * sizeof(double) );
	#pragma acc enter data create(CsrCofA[0:NonzeroCountA])
	#pragma acc enter data create(CsrIndA[0:NonzeroCountA])
	#pragma acc enter data create(VectorB[0:N])
	
    #pragma acc kernels present(CsrPtrA[0:N],CsrCofA[0:NonzeroCountA],CsrIndA[0:NonzeroCountA])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<fluidcount;++iP){
		#pragma acc loop independent
		for(int jN=0;jN<NeighborFluidCount[iP];++jN){
			const int jP = Neighbor[iP][jN];
			#pragma acc loop seq
			for(int rD=0;rD<DIM;++rD){
				#pragma acc loop seq
				for(int sD=0;sD<DIM;++sD){
					const int iRow    = DIM*iP+rD;
					const int jColumn = DIM*jP+sD;
					const int iNonzero = CsrPtrA[iRow]+DIM*jN+sD;
					CsrIndA[ iNonzero ] = jColumn;
				}
			}
		}
	}
	
    #pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iNonzero=0;iNonzero<NonzeroCountA;++iNonzero){
		CsrCofA[ iNonzero ] = 0.0;
	}
	
    #pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<N;++iRow){
		VectorB[iRow]=0.0;
	}
	
    #pragma acc kernels present(Property[0:ParticleCount],r[0:ParticleCount][0:DIM],v[0:ParticleCount][0:DIM],m[0:ParticleCount],Mu[0:ParticleCount],CsrCofA[0:NonzeroCountA],CsrIndA[0:NonzeroCountA],CsrPtrA[0:N],VectorB[0:N])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=FluidParticleBegin;iP<FluidParticleEnd;++iP){
		
		// Viscosity term
		int iN;
		double selfCof[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		double sumvec[DIM]={0.0,0.0,0.0};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCount[iP];++jN){
			const int jP = Neighbor[iP][jN];
			if(iP==jP){
				iN=jN;
				continue;
			}
			double rij[DIM];
			#pragma acc loop seq
			for(int rD=0;rD<DIM;++rD){
				rij[rD] =  Mod(r[jP][rD] - r[iP][rD] + 0.5*DomainWidth[rD], DomainWidth[rD]) -0.5*DomainWidth[rD];
			}
			const double rij2 = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2];
			if(RadiusV*RadiusV -rij2 > 0){
				const double dij = sqrt(rij2);
				const double wdrij = -dwvdr(dij,RadiusV);
				const double eij[DIM] = {rij[0]/dij,rij[1]/dij,rij[2]/dij};
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
				
				#pragma acc loop seq
				for(int rD=0;rD<DIM;++rD){
					const int iRow = DIM*iP+rD;
					#pragma acc loop seq
					for(int sD=0;sD<DIM;++sD){
						#ifdef TWO_DIMENSIONAL
						const double coefficient = +8.0*muij*eij[sD]*eij[rD]*wdrij/dij;
						#else
						const double coefficient = +10.0*muij*eij[sD]*eij[rD]*wdrij/dij;
						#endif
						
						selfCof[rD][sD]+=coefficient;
						
						if(FLUID_BEGIN<=Property[jP] && Property[jP]<FLUID_END){
							const int jColumn = DIM*jP+sD;
							const int jNonzero= CsrPtrA[iRow]+DIM*jN+sD;
							// assert( CsrIndA [ jNonzero ] == jColumn);
							CsrCofA [ jNonzero ] = -coefficient;
						}
						else if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
							sumvec[rD] += coefficient*v[jP][sD];
						}
					}
				}
			}
		}
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow = DIM*iP+rD;
			#pragma acc loop seq
			for(int sD=0;sD<DIM;++sD){
				const int iColumn = DIM*iP+sD;
				const int iNonzero= CsrPtrA[iRow]+DIM*iN+sD;
				// assert( CsrIndA[ iNonzero ] == iColumn);
				CsrCofA[ iNonzero ] = selfCof[rD][sD];
			}
		}
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow = DIM*iP+rD;
			VectorB[iRow]+=sumvec[rD];
		}
		
		// Ineritial Force term
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			const int iRow = DIM*iP+iD;
			const int iColumn = DIM*iP+iD;
			const int iNonzero= CsrPtrA[iRow]+DIM*iN+iD;
			const double coefficient =  m[iP]/ParticleVolume / Dt;
			// assert( CsrIndA[ iNonzero ] == iColumn );
			CsrCofA[ iNonzero ] += coefficient;
			VectorB[iRow] += coefficient*v[iP][iD];
		}
	}    
}

static int    NonzeroCountC;
static double *CsrCofC; // [ FluidCount * DIM x NeighCount ]
static int    *CsrIndC; // [ FluidCount * DIM x NeighCount ]
static int    *CsrPtrC; // [ FluidCount * DIM + 1 ] NeighCount‚ÌÏŽZ—ñ
static double *VectorP; // [ ParticleCount ]
#pragma acc declare create(NonzeroCountC, CsrCofC,CsrIndC,CsrPtrC,VectorP)

static void calculateMatrixC( void )
{
	const double (*r)[DIM] = Position;
    const double (*v)[DIM] = Velocity;
	
	// Copy DIM*NeighborCountP to CsrPtrC
	int power = 0;
	const int fluidcount = FluidParticleEnd-FluidParticleBegin;
	const int N = DIM*(fluidcount);
	while( (N>>power) != 0 ){
		power++;
	}
	const int powerN = (1<<power);
	
	CsrPtrC = (int *)malloc( powerN * sizeof(int));
	#pragma acc enter data create(CsrPtrC[0:powerN])
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<powerN;++iRow){
		CsrPtrC[iRow]=0;
	}
	
	#pragma acc kernels present(CsrPtrC[0:powerN])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<fluidcount;++iP){
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow=DIM*iP+rD+1;
			CsrPtrC[iRow] = NeighborCountP[iP];
		}
	}
	
	// Convert CsrPtrC to cumulative sum
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (1<<iMain);	
		#pragma acc kernels present(CsrPtrC[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			CsrPtrC[iRow]+=CsrPtrC[iRow+dist];
		}
	}
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (powerN>>(iMain+1));	
		#pragma acc kernels present(CsrPtrC[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			CsrPtrC[iRow]-=CsrPtrC[iRow+dist];
			CsrPtrC[iRow+dist]+=CsrPtrC[iRow];
		}
	}
	
	#pragma acc kernels present(CsrPtrC[0:powerN])
	#pragma acc loop seq
	for(int iDummy=0;iDummy<1;++iDummy){
		NonzeroCountC = CsrPtrC[N];
	}
	#pragma acc update host(NonzeroCountC)
	
	// calculate coefficient matrix C and source vector P
	CsrCofC = (double *)malloc( NonzeroCountC * sizeof(double) );
	CsrIndC = (int *)malloc( NonzeroCountC * sizeof(int) );
	VectorP = (double *)malloc( ParticleCount * sizeof(double) );
	#pragma acc enter data create(CsrCofC[0:NonzeroCountC])
	#pragma acc enter data create(CsrIndC[0:NonzeroCountC])
	#pragma acc enter data create(VectorP[0:ParticleCount])
	
	#pragma acc kernels present(CsrPtrC[0:powerN],CsrIndC[0:NonzeroCountC],CsrCofC[0:NonzeroCountC],)
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<fluidcount;++iP){
		#pragma acc loop independent
		for(int jN=0;jN<NeighborCountP[iP];++jN){
			const int jP = NeighborP[iP][jN];
			#pragma acc loop seq
			for(int rD=0;rD<DIM;++rD){
				const int iRow    = DIM*iP+rD;
				const int iColumn = jP;
				const int iNonzero = CsrPtrC[iRow]+jN;
				CsrIndC[ iNonzero ] = iColumn;
			}
		}
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iNonzero=0;iNonzero<NonzeroCountC;++iNonzero){
		CsrCofC[ iNonzero ] = 0.0;
	}
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		VectorP[iP] = 0.0;
	}
	
	// set matrix C
	#pragma acc kernels present(r[0:ParticleCount][0:DIM],CsrCofC[0:NonzeroCountC],CsrIndC[0:NonzeroCountC],CsrPtrC[0:N])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<fluidcount;++iP){
		int iN;
		double selfCof[DIM] = {0.0,0.0,0.0};
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCountP[iP];++jN){
			const int jP = NeighborP[iP][jN];
			if(iP==jP){
				iN=jN;
				continue;
			}
			double rij[DIM];
			#pragma acc loop seq
			for(int rD=0;rD<DIM;++rD){
				rij[rD] =  Mod(r[jP][rD] - r[iP][rD] + 0.5*DomainWidth[rD], DomainWidth[rD]) -0.5*DomainWidth[rD];
			}
			const double rij2 = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2];
			if(RadiusP*RadiusP -rij2 > 0){
				const double dij = sqrt(rij2);
				const double eij[DIM] = {rij[0]/dij,rij[1]/dij,rij[2]/dij};
				const double wpdrij = -dwpdr(dij,RadiusP);
				#pragma acc loop seq
				for(int rD=0;rD<DIM;++rD){
					const int iRow = DIM*iP+rD;
					const double coefficient = eij[rD]*wpdrij;
					selfCof[rD]+=coefficient;
					const int jColumn = jP;
					const int iNonzero = CsrPtrC[iRow]+jN;
					//assert( CsrIndC[ iNonzero ] == jColumn );
					CsrCofC[ iNonzero ] = coefficient;
				}
			}
		}
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow = DIM*iP+rD;
			const int iColumn = iP;
			const int iNonzero = CsrPtrC[iRow]+iN;
			//assert( CsrIndC[ iNonzero ] == iColumn );
			CsrCofC[ iNonzero ] = selfCof[rD];
		}
	}

	// set vector P
	#pragma acc kernels present(Property[0:ParticleCount],r[0:ParticleCount][0:DIM],v[0:ParticleCount][0:DIM],Lambda[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){			
		
		VectorP[iP]=0.0;
		if(VolStrainP[iP]>0.0){
			VectorP[iP] = Kappa[iP]*VolStrainP[iP];
		}
		double sum = 0.0;
		#pragma acc loop seq
		for(int jN=0;jN<NeighborCountP[iP];++jN){
			const int jP = NeighborP[iP][jN];
			if(iP==jP)continue;
			double rij[DIM];
			#pragma acc loop seq
			for(int rD=0;rD<DIM;++rD){
				rij[rD] =  Mod(r[jP][rD] - r[iP][rD] + 0.5*DomainWidth[rD], DomainWidth[rD]) -0.5*DomainWidth[rD];
			}
			const double rij2 = rij[0]*rij[0]+rij[1]*rij[1]+rij[2]*rij[2];
			if(RadiusP*RadiusP -rij2 > 0){
				const double dij = sqrt(rij2);
				const double eij[DIM] = {rij[0]/dij,rij[1]/dij,rij[2]/dij};
				const double wpdrij = -dwpdr(dij,RadiusP);
				#pragma acc loop seq
				for(int sD=0;sD<DIM;++sD){
					if(WALL_BEGIN<=Property[iP] && Property[iP]<WALL_END){
						const double coefficient = +eij[sD]*wpdrij;
						sum += Lambda[iP]*coefficient*v[iP][sD];
					}
					if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END){
						const double coefficient = -eij[sD]*wpdrij;
						sum += Lambda[iP]*coefficient*v[jP][sD];
					}
				}
			}
		}
		VectorP[iP]+=sum;
	}

}

static void multiplyMatrixC( void )
{
	const int fluidcount = FluidParticleEnd-FluidParticleBegin;
	const int N = DIM*(fluidcount);
	const int M = ParticleCount;
	
	int power = 0;
	while( (N>>power) != 0 ){
		power++;
	}
	const int powerN = (1<<power);
	
	// b = b - Cp
	#pragma acc kernels present(CsrPtrC[0:powerN],CsrIndC[0:NonzeroCountC],CsrCofC[0:NonzeroCountC],VectorP[0:M])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<N;++iRow){
		#pragma acc loop seq
		for(int iNonzero=CsrPtrC[iRow];iNonzero<CsrPtrC[iRow+1];++iNonzero){
			const int iColumn = CsrIndC[iNonzero];
			VectorB[iRow] -= CsrCofC[iNonzero] * VectorP[iColumn];
		}
	}
	
	// A = A + Cƒ©C^T
	#pragma acc kernels present(CsrPtrA[0:N],CsrIndA[0:NonzeroCountA],CsrCofA[0:NonzeroCountA],CsrPtrC[0:N],CsrIndC[0:NonzeroCountC],CsrCofC[0:NonzeroCountC],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT],NeighborCountP[0:ParticleCount],Lambda[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<fluidcount;++iP){
		#pragma acc loop independent
		for(int jN=0;jN<NeighborFluidCount[iP];++jN){
			const int jP = Neighbor[iP][jN];
			int iNeigh=0;
			int jNeigh=0;
			double sum[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
			#pragma acc loop seq
			while(iNeigh<NeighborCountP[iP] && jNeigh<NeighborCountP[jP]){
				const int iNP = NeighborP[iP][iNeigh];
				const int jNP = NeighborP[jP][jNeigh];
				if(iNP==jNP){
					#pragma acc loop seq
					for(int rD=0;rD<DIM;++rD){
						#pragma acc loop seq
						for(int sD=0;sD<DIM;++sD){
							const int iRowC    = DIM*iP+rD;
							const int iColumnC = iNP;
							const int iNonzeroC= CsrPtrC[iRowC]+iNeigh;
							// assert(CsrIndC[iNonzeroC]==iColumnC);
							const int jRowC    = DIM*jP+sD;
							const int jColumnC = jNP;
							const int jNonzeroC= CsrPtrC[jRowC]+jNeigh;
							// assert(CsrIndC[jNonzeroC]==jColumnC);
							sum[rD][sD] += CsrCofC[ iNonzeroC ] * Lambda[iNP] * CsrCofC[ jNonzeroC ];
						}
					}
					iNeigh++;
					jNeigh++;
				}
				else if(iNP<jNP){
					iNeigh++;
				}
				else if(iNP>jNP){
					jNeigh++;
				}
				else{
					break;
				}
			}
			#pragma acc loop seq
			for(int rD=0;rD<DIM;++rD){
				#pragma acc loop seq
				for(int sD=0;sD<DIM;++sD){
					const int iRowA    = DIM*iP+rD;
					const int iColumnA = DIM*jP+sD;
					const int iNonzeroA= CsrPtrA[iRowA]+DIM*jN+sD;
					// assert(CsrIndA[ iNonzeroA ]==iColumnA);
					CsrCofA[iNonzeroA] += sum[rD][sD];
				}
			}
		}
	}
	
	free(CsrCofC);
	free(CsrIndC);
	free(CsrPtrC);
	free(VectorP);
	#pragma acc exit data delete(CsrPtrC,CsrIndC,CsrCofC,VectorP)

}

static void myDcsrmv( const int m, const int nnz, const double alpha, const double *csrVal, const int *csrRowPtr, const int *csrColInd, const double *x, const double beta, double *y)
{
	#pragma acc kernels present(csrVal[0:nnz],csrRowPtr[0:m+1],csrColInd[0:nnz],x[0:m],y[0:m])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<m; ++iRow){
		double sum = 0.0;
		#pragma acc loop reduction(+:sum) vector
		for(int iNonZero=csrRowPtr[iRow];iNonZero<csrRowPtr[iRow+1];++iNonZero){
			const int iColumn=csrColInd[iNonZero];
			sum += alpha*csrVal[iNonZero]*x[iColumn];
		}
		y[iRow] *= beta;
		y[iRow] += sum;
	}
}

static void myDdot( const int n, const double *x, const double *y, double *res )
{
	double sum=0.0;
	#pragma acc kernels copy(sum) present(x[0:n],y[0:n])
	#pragma acc loop reduction(+:sum)
	#pragma omp parallel for reduction(+:sum)
	for(int iRow=0;iRow<n;++iRow){
		sum += x[iRow]*y[iRow];
	}
	(*res)=sum;
}

static void myDaxpy( const int n, const double alpha, const double *x, double *y )
{
	#pragma acc kernels present(x[0:n],y[0:n])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<n;++iRow){
		y[iRow] += alpha*x[iRow];
	}
}

static void myDcopy( const int n, const double *x, double *y )
{
	#pragma acc kernels present(x[0:n],y[0:n])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<n;++iRow){
		y[iRow] = x[iRow];
	}
}


static void solveWithConjugatedGradient(void){
	const int fluidcount = (FluidParticleEnd-FluidParticleBegin);
	const int N = DIM*fluidcount;
	
	const double *b = VectorB;
	double *x = (double *)malloc( N*sizeof(double) );
	double *r = (double *)malloc( N*sizeof(double) );
	double *z = (double *)malloc( N*sizeof(double) );
	double *p = (double *)malloc( N*sizeof(double) );
	double *q = (double *)malloc( N*sizeof(double) );
	double rho=0.0;
	double rhop=0.0;
	double tmp=0.0;
	double alpha=0.0;
	double beta=0.0;
	double nrm=0.0;
	double nrm0=0.0;
	int iter=0;
	
	#pragma acc enter data create(x[0:N],r[0:N],z[0:N],p[0:N],q[0:N])

	// intialize
	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM],x[0:N],Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<fluidcount;++iP){
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow = DIM*iP+rD;
			x[iRow]=Velocity[iP][rD]-Force[iP][rD]/Mass[iP]*Dt;
		}
	}
		
	myDcopy( N, b, r );	
	myDcsrmv( N, NonzeroCountA, -1.0, CsrCofA, CsrPtrA, CsrIndA, x, 1.0, r );
	myDdot( N, b, b, &nrm0 );
	nrm0=sqrt(nrm0);
	myDdot( N, r, r, &nrm );
	nrm=sqrt(nrm);
	
	if(nrm!=0.0)for(iter=0;iter<N;++iter){
		myDcopy( N, r, z );
		rhop = rho;
		myDdot( N, r, z, &rho);
		if(iter==0){
			myDcopy( N, z, p );
		}
		else{
			beta=rho/rhop;
			myDaxpy( N, beta, p, z );
			myDcopy( N, z, p );
		}
		myDcsrmv( N, NonzeroCountA, 1.0, CsrCofA, CsrPtrA, CsrIndA, p, 0.0, q);
		myDdot( N, p, q, &tmp );
		alpha =rho/tmp;
		myDaxpy( N, alpha, p, x );
		myDaxpy( N,-alpha, q, r );
		myDdot( N, r, r, &nrm );
		nrm=sqrt(nrm);
		
		if(nrm/nrm0 < 1.0e-7 )break;
		
	}
	
	log_printf("nrm=%e, nrm0=%e, iter=%d\n",nrm,nrm0,iter);
//	myDcopy( N, b, z );	
//	myDcsrmv( N, N, -1.0, CsrCofA, CsrPtrA, CsrIndA, x, 1.0, z );
//	myDdot( N, z, z, &nrm );
//	nrm=sqrt(nrm);
//	fprintf(stderr,"check nrm=%e\n",nrm);
	
	
	//copy to Velocity
	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM],x[0:N])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<fluidcount;++iP){
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow = DIM*iP+rD;
			Velocity[iP][rD]=x[iRow];
		}
	}
	
	
	free(x);
	free(r);
	free(z);
	free(p);
	free(q);
	#pragma acc exit data delete(x[0:N],r[0:N],z[0:N],p[0:N],q[0:N])

	
	free(CsrCofA);
	free(CsrIndA);
	free(CsrPtrA);
	free(VectorB);
	#pragma acc exit data delete(CsrCofA,CsrIndA,CsrPtrA,VectorB)

		
}


static void calculateVirialPressureAtParticle()
{
    const double (*x)[DIM] = Position;
    const double (*p) = PressureP;

	#pragma acc kernels present(x[0:ParticleCount][0:DIM],p[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	double virialAtParticle=0.0;
    	#pragma acc loop seq
        for(int iN=0;iN<NeighborCountP[iP];++iN){
            const int jP=NeighborP[iP][iN];
        	if(iP==jP)continue;
            double xij[DIM];
        	#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(RadiusP*RadiusP - rij2 > 0){
                const double rij = sqrt(rij2);
                const double dwij = dwpdr(rij,RadiusP);
                virialAtParticle -= 0.5*(p[iP]+p[jP])*dwij*rij*ParticleVolume;
            }
        }
#ifdef TWO_DIMENSIONAL
        VirialPressureAtParticle[iP] = 1.0/2.0/ParticleVolume * virialAtParticle;
#else
    	VirialPressureAtParticle[iP] = 1.0/3.0/ParticleVolume * virialAtParticle;
#endif
    }
}

static void calculateVirialPressureInsideRadius()
{
	const double (*x)[DIM] = Position;
	
	#pragma acc kernels present(Property[0:ParticleCount],x[0:ParticleCount][0:DIM],VirialPressureAtParticle[0:ParticleCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		int count=1;
		double sum = VirialPressureAtParticle[iP];
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCountP[iP];++iN){
            const int jP=NeighborP[iP][iN];
			if(iP==jP)continue;
			if(WALL_BEGIN<=Property[jP] && Property[jP]<WALL_END)continue;
            double xij[DIM];
			#pragma acc loop seq
            for(int iD=0;iD<DIM;++iD){
                xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
            }
            const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
            if(RadiusP*RadiusP - rij2 > 0){
            	count +=1;
                sum += VirialPressureAtParticle[jP];
            }
        }
		VirialPressureInsideRadius[iP] = sum/count;
	}
}


static void calculateVirialStressAtParticle()
{
	const double (*x)[DIM] = Position;
	const double (*v)[DIM] = Velocity;
	

	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD]=0.0;
			}
		}
	}
	
	#pragma acc kernels present(x[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCountP[iP];++iN){
			const int jP=NeighborP[iP][iN];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			// pressureP
			if(RadiusP*RadiusP - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = dwpdr(rij,RadiusP);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureP[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],x[0:ParticleCount][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// pressureA
			if(RadiusA*RadiusA - rij2 > 0){
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dwij = ratio * dwadr(rij,RadiusA);
				double gradw[DIM] = {dwij*xij[0]/rij,dwij*xij[1]/rij,dwij*xij[2]/rij};
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = (PressureA[iP])*gradw[iD]*ParticleVolume;
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}

	}
	
	#pragma acc kernels present(x[0:ParticleCount][0:DIM],v[0:ParticleCount][0:DIM],Mu[0:ParticleCount])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// viscosity term
			if(RadiusV*RadiusV - rij2 > 0){
				const double rij = sqrt(rij2);
				const double dwij = -dwvdr(rij,RadiusV);
				const double eij[DIM] = {xij[0]/rij,xij[1]/rij,xij[2]/rij};
				const double vij[DIM] = {v[jP][0]-v[iP][0],v[jP][1]-v[iP][1],v[jP][2]-v[iP][2]};
				const double muij = 2.0*(Mu[iP]*Mu[jP])/(Mu[iP]+Mu[jP]);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#ifdef TWO_DIMENSIONAL
					fij[iD] = 8.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#else
					fij[iD] = 10.0*muij*(vij[0]*eij[0]+vij[1]*eij[1]+vij[2]*eij[2])*eij[iD]*dwij/rij*ParticleVolume;
					#endif
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=0.5*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}
	
	#pragma acc kernels present(Property[0:ParticleCount],x[0:ParticleCount][0:DIM])
	#pragma acc loop independent	
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		double stress[DIM][DIM]={{0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0}};
		#pragma acc loop seq
		for(int iN=0;iN<NeighborCount[iP];++iN){
			const int jP=Neighbor[iP][iN];
			if(iP==jP)continue;
			double xij[DIM];
			#pragma acc loop seq
			for(int iD=0;iD<DIM;++iD){
				xij[iD] = Mod(x[jP][iD] - x[iP][iD] +0.5*DomainWidth[iD] , DomainWidth[iD]) -0.5*DomainWidth[iD];
			}
			const double rij2 = (xij[0]*xij[0] + xij[1]*xij[1] + xij[2]*xij[2]);
			
			
			// diffuse interface force (1st term)
			if(RadiusG*RadiusG - rij2 > 0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double weight = ratio * wg(rij,RadiusG);
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*( -GravityCenter[iP][iD])*weight/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
			
			// diffuse interface force (2nd term)
			if(RadiusG*RadiusG - rij2 > 0.0){
				const double a = CofA[Property[iP]]*(CofK)*(CofK);
				double ratio = InteractionRatio[Property[iP]][Property[jP]];
				const double rij = sqrt(rij2);
				const double dw = ratio * dwgdr(rij,RadiusG);
				const double gradw[DIM] = {dw*xij[0]/rij,dw*xij[1]/rij,dw*xij[2]/rij};
				double gr=0.0;
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					gr += (                     -GravityCenter[iP][iD])*xij[iD];
				}
				double fij[DIM] = {0.0,0.0,0.0};
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					fij[iD] = -a*(gr)*gradw[iD]/R2g*RadiusG * (ParticleVolume/ParticleSpacing);
				}
				#pragma acc loop seq
				for(int iD=0;iD<DIM;++iD){
					#pragma acc loop seq
					for(int jD=0;jD<DIM;++jD){
						stress[iD][jD]+=1.0*fij[iD]*xij[jD]/ParticleVolume;
					}
				}
			}
		}
		#pragma acc loop seq
		for(int iD=0;iD<DIM;++iD){
			#pragma acc loop seq
			for(int jD=0;jD<DIM;++jD){
				VirialStressAtParticle[iP][iD][jD] += stress[iD][jD];
			}
		}
	}	
	
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		#ifdef TWO_DIMENSIONAL
		VirialPressureAtParticle[iP]=-1.0/2.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]);
		#else
		VirialPressureAtParticle[iP]=-1.0/3.0*(VirialStressAtParticle[iP][0][0]+VirialStressAtParticle[iP][1][1]+VirialStressAtParticle[iP][2][2]);
		#endif
	}

}


static void calculatePeriodicBoundary( void )
{
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
    for(int iP=0;iP<ParticleCount;++iP){
    	#pragma acc loop seq
        for(int iD=0;iD<DIM;++iD){
            Position[iP][iD] = Mod(Position[iP][iD]-DomainMin[iD],DomainWidth[iD])+DomainMin[iD];
        }
    }
}

