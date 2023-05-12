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

#ifdef _CUDA
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#endif

#include "errorfunc.h"
#include "log.h"

const double DOUBLE_ZERO[32]={0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0,
                              0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0};

using namespace std;
//#define TWO_DIMENSIONAL
//#define CONVERGENCE_CHECK

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
static void freeMatrixA( void );
static void calculateMatrixC( void );
static void freeMatrixC( void );
static void multiplyMatrixC( void );
static void calculateMultiGridMatrix( void );
static void freeMultiGridMatrix( void );
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
	clock_t cNeigh=0, cExplicit=0, cImplicitTriplet=0, cImplicitInsert=0, cImplicitMatrix=0, cImplicitMulti=0, cImplicitSolve=0, cPrecondition=0, cVirial=0, cOther=0;

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
	
//	#pragma acc enter data create(ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
//	#pragma acc enter data create(ParticleCount,ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount])
//	#pragma acc enter data create(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
//	#pragma acc enter data create(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
//	#pragma acc enter data create(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
//	#pragma acc enter data create(NeighborFluidCount[0:ParticleCount],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
//	#pragma acc enter data create(NeighborCountP[0:ParticleCount],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
//	#pragma acc enter data create(TmpIntScalar[0:ParticleCount],TmpDoubleScalar[0:ParticleCount],TmpDoubleVector[0:ParticleCount][0:DIM])
//	#pragma acc enter data create(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],TotalCellCount)
//	#pragma acc enter data create(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
//	#pragma acc enter data create(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
//	#pragma acc enter data create(FluidParticleBegin,FluidParticleEnd)
//	#pragma acc enter data create(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
//	#pragma acc enter data create(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
//	#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount],VirialPressureInsideRadius[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
//	#pragma acc enter data create(Lambda[0:ParticleCount],Kappa[0:ParticleCount],Mu[0:ParticleCount])
//	#pragma acc enter data create(Gravity[0:DIM])
//	#pragma acc enter data create(WallParticleBegin,WallParticleEnd)
//	#pragma acc enter data create(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
//	#pragma acc enter data create(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	
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
	OutputNext = Time;
	VtkOutputNext = Time;
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
		freeMatrixC();
		cTill = clock(); cImplicitMulti += (cTill-cFrom); cFrom = cTill;
		calculateMultiGridMatrix();
		cTill = clock(); cPrecondition += (cTill-cFrom); cFrom = cTill;
		solveWithConjugatedGradient();
		freeMultiGridMatrix();
		freeMatrixA();
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
    	log_printf("precondition             %lf [CPU sec]\n", (double)cPrecondition/CLOCKS_PER_SEC);
    	log_printf("implicit solve         : %lf [CPU sec]\n", (double)(cPrecondition+cImplicitSolve)/CLOCKS_PER_SEC);
    	log_printf("virial calculation:      %lf [CPU sec]\n", (double)cVirial/CLOCKS_PER_SEC);
    	log_printf("other calculation:       %lf [CPU sec]\n", (double)cOther/CLOCKS_PER_SEC);
    	log_printf("total:                   %lf [CPU sec]\n", (double)(cNeigh+cExplicit+cImplicitTriplet+cImplicitInsert+cImplicitMatrix+cImplicitMulti+cPrecondition+cImplicitSolve+cVirial+cOther)/CLOCKS_PER_SEC);
    	log_printf("total (check):           %lf [CPU sec]\n", (double)(cEnd-cStart)/CLOCKS_PER_SEC);
    }

	
//	���I�������m�ۂ����Ă��Ȃ��ϐ���z��ɑ΂��ă�����������s����cuMemFree���G���[��Ԃ�
//	�v���O�����I�����̃���������Ɋ��҂��āA�����I�ȃ���������͏ȗ�
	
//	#pragma acc exit data delete(ParticleSpacing,ParticleVolume,Dt,DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
//	#pragma acc exit data delete(ParticleCount,ParticleIndex[0:ParticleCount],Property[0:ParticleCount],Mass[0:ParticleCount])
//	#pragma acc exit data delete(Density[0:TYPE_COUNT],BulkModulus[0:TYPE_COUNT],BulkViscosity[0:TYPE_COUNT],ShearViscosity[0:TYPE_COUNT],SurfaceTension[0:TYPE_COUNT])
//	#pragma acc exit data delete(CofA[0:TYPE_COUNT],CofK,InteractionRatio[0:TYPE_COUNT][0:TYPE_COUNT])
//	#pragma acc exit data delete(Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM])
//	#pragma acc exit data delete(NeighborFluidCount[0:ParticleCount],NeighborCount[0:ParticleCount],Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
//	#pragma acc exit data delete(NeighborCountP[0:ParticleCount],NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT])
//	#pragma acc exit data delete(TmpIntScalar[0:ParticleCount],TmpDoubleScalar[0:ParticleCount],TmpDoubleVector[0:ParticleCount][0:DIM])
//	#pragma acc exit data delete(PowerParticleCount,ParticleCountPower,CellWidth,CellCount[0:DIM],TotalCellCount)
//	#pragma acc exit data delete(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
//	#pragma acc exit data delete(CellIndex[0:PowerParticleCount],CellParticle[0:PowerParticleCount])
//	#pragma acc exit data delete(FluidParticleBegin,FluidParticleEnd)
//	#pragma acc exit data delete(DensityA[0:ParticleCount],GravityCenter[0:ParticleCount][0:DIM],PressureA[0:ParticleCount])
//	#pragma acc exit data delete(VolStrainP[0:ParticleCount],DivergenceP[0:ParticleCount],PressureP[0:ParticleCount])
//	#pragma acc exit data delete(VirialPressureAtParticle[0:ParticleCount],VirialPressureInsideRadius[0:ParticleCount],VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM])
//	#pragma acc exit data delete(Lambda[0:ParticleCount],Kappa[0:ParticleCount],Mu[0:ParticleCount])
//	#pragma acc exit data delete(Gravity[0:DIM])
//	#pragma acc exit data delete(WallParticleBegin,WallParticleEnd)
//	#pragma acc exit data delete(WallCenter[0:WALL_END][0:DIM],WallVelocity[0:WALL_END][0:DIM],WallOmega[0:WALL_END][0:DIM],WallRotation[0:WALL_END][0:DIM][0:DIM])
//	#pragma acc exit data delete(MaxRadius,RadiusA,RadiusG,RadiusP,RadiusV,Swa,Swg,Swp,Swv,N0a,N0p,R2g)
	
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
    	
    	#pragma acc enter data create(ParticleIndex[0:ParticleCount]) attach(ParticleIndex)
    	#pragma acc enter data create(Property[0:ParticleCount]) attach(Property)
    	#pragma acc enter data create(Position[0:ParticleCount][0:DIM]) attach(Position)
    	#pragma acc enter data create(Velocity[0:ParticleCount][0:DIM]) attach(Velocity)
    	#pragma acc enter data create(DensityA[0:ParticleCount]) attach(DensityA)
    	#pragma acc enter data create(GravityCenter[0:ParticleCount][0:DIM]) attach(GravityCenter)
    	#pragma acc enter data create(PressureA[0:ParticleCount]) attach(PressureA)
    	#pragma acc enter data create(VolStrainP[0:ParticleCount]) attach(VolStrainP)
    	#pragma acc enter data create(DivergenceP[0:ParticleCount]) attach(DivergenceP)
   		#pragma acc enter data create(PressureP[0:ParticleCount]) attach(PressureP)
    	#pragma acc enter data create(VirialPressureAtParticle[0:ParticleCount]) attach(VirialPressureAtParticle)
    	#pragma acc enter data create(VirialPressureInsideRadius[0:ParticleCount]) attach(VirialPressureInsideRadius)
    	#pragma acc enter data create(VirialStressAtParticle[0:ParticleCount][0:DIM][0:DIM]) attach(VirialStressAtParticle)
    	#pragma acc enter data create(Mass[0:ParticleCount]) attach(Mass)
    	#pragma acc enter data create(Force[0:ParticleCount][0:DIM]) attach(Force)
    	#pragma acc enter data create(Mu[0:ParticleCount]) attach(Mu)
    	#pragma acc enter data create(Lambda[0:ParticleCount]) attach(Lambda)
    	#pragma acc enter data create(Kappa[0:ParticleCount]) attach(Kappa)
    	
    	#pragma acc enter data create(TmpIntScalar[0:ParticleCount]) attach(TmpIntScalar)
    	#pragma acc enter data create(TmpDoubleScalar[0:ParticleCount]) attach(TmpDoubleScalar)
    	#pragma acc enter data create(TmpDoubleVector[0:ParticleCount][0:DIM]) attach(TmpDoubleVector)
    	
    	#pragma acc enter data create(NeighborFluidCount[0:ParticleCount]) attach(NeighborFluidCount)
    	#pragma acc enter data create(NeighborCount[0:ParticleCount]) attach(NeighborCount)
    	#pragma acc enter data create(Neighbor[0:ParticleCount][0:MAX_NEIGHBOR_COUNT]) attach(Neighbor)
    	#pragma acc enter data create(NeighborCountP[0:ParticleCount]) attach(NeighborCountP)
    	#pragma acc enter data create(NeighborP[0:ParticleCount][0:MAX_NEIGHBOR_COUNT]) attach(NeighborP)

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
	
	#pragma acc update device(ParticleCount,ParticleSpacing,DomainMin[0:DIM],DomainMax[0:DIM],ParticleVolume)
	#pragma acc update device(Property[0:ParticleCount],Position[0:ParticleCount][0:DIM],Velocity[0:ParticleCount][0:DIM])
	#pragma acc update device(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
	
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
	fprintf(fp, "SCALARS VolStrainP float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "%e\n", (float)VolStrainP[iP]);
	}
	fprintf(fp, "\n");
	fprintf(fp, "SCALARS DivergenceP float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "%e\n", (float)DivergenceP[iP]);
	}
	fprintf(fp, "\n");
	fprintf(fp, "SCALARS PressureP float 1\n");
	fprintf(fp, "LOOKUP_TABLE default\n");
	for(int iP=0;iP<ParticleCount;++iP){
		fprintf(fp, "%e\n", (float)PressureP[iP]);
	}
	fprintf(fp, "\n");
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
		const int range = (int)ceil(radius_ratio);
		const int rangeX = range;
		const int rangeY = range;
		#ifdef TWO_DIMENSIONAL
		const int rangeZ = 0;
		#else
		const int rangeZ = range;
		#endif
		
		int count = 0;
		double sum = 0.0;
		for(int iX=-rangeX;iX<=rangeX;++iX){
			for(int iY=-rangeY;iY<=rangeY;++iY){
				for(int iZ=-rangeZ;iZ<=rangeZ;++iZ){
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
		
		N0a = sum;
		log_printf("N0a = %e, count=%d\n", N0a, count);
	}	
	
    {// N0p
        const double radius_ratio = RadiusP/ParticleSpacing;
        const int range = (int)ceil(radius_ratio);
    	const int rangeX = range;
		const int rangeY = range;
		#ifdef TWO_DIMENSIONAL
		const int rangeZ = 0;
		#else
		const int rangeZ = range;
		#endif

        int count = 0;
        double sum = 0.0;
        for(int iX=-rangeX;iX<=rangeX;++iX){
            for(int iY=-rangeY;iY<=rangeY;++iY){
                for(int iZ=-rangeZ;iZ<=rangeZ;++iZ){
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
        N0p = sum;
        log_printf("N0p = %e, count=%d\n", N0p, count);
    }
	
	#pragma acc update device(RadiusA,RadiusG,RadiusP,RadiusV)
	#pragma acc update device(Swa,Swg,Swp,Swv,R2g)
	#pragma acc update device(N0a,N0p)
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
	
	#pragma acc update device(Mass[0:ParticleCount])
	#pragma acc update device(Kappa[0:ParticleCount])
	#pragma acc update device(Lambda[0:ParticleCount])
	#pragma acc update device(Mu[0:ParticleCount])
	#pragma acc update device(CofK,CofA[0:TYPE_COUNT])
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
	
	#pragma acc update device(WallRotation[0:WALL_END][0:DIM][0:DIM])

}

static void initializeDomain( void )
{
	MaxRadius = ((RadiusA>MaxRadius) ? RadiusA : MaxRadius);
	MaxRadius = ((2.0*RadiusP>MaxRadius) ? 2.0*RadiusP : MaxRadius);
	MaxRadius = ((RadiusV>MaxRadius) ? RadiusV : MaxRadius);
	fprintf(stderr, "MaxRadius = %lf\n", MaxRadius);
	
	//CellWidth=ParticleSpacing;
	CellWidth = ceil(MaxRadius/ParticleSpacing)*ParticleSpacing;
	
	double cellCount[DIM];
	
	cellCount[0] = (DomainMax[0] - DomainMin[0])/CellWidth;
	cellCount[1] = (DomainMax[1] - DomainMin[1])/CellWidth;
	#ifdef TWO_DIMENSIONAL
	cellCount[2] = 1;
	#else
	cellCount[2] = (DomainMax[2] - DomainMin[2])/CellWidth;
	#endif
	
	CellCount[0] = (int)ceil(cellCount[0]);
	CellCount[1] = (int)ceil(cellCount[1]);
	CellCount[2] = (int)ceil(cellCount[2]);
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
	#pragma acc enter data create(CellFluidParticleBegin[0:TotalCellCount]) attach(CellFluidParticleBegin)
	#pragma acc enter data create(CellFluidParticleEnd[0:TotalCellCount]) attach(CellFluidParticleEnd)
	#pragma acc enter data create(CellWallParticleBegin[0:TotalCellCount]) attach(CellWallParticleBegin)
	#pragma acc enter data create(CellWallParticleEnd[0:TotalCellCount]) attach(CellWallParticleEnd)
	
	// calculate minimun PowerParticleCount which sataisfies  ParticleCount < PowerParticleCount = pow(2,ParticleCountPower) 
	ParticleCountPower=0;
	while((ParticleCount>>ParticleCountPower)!=0){
		++ParticleCountPower;
	}
	PowerParticleCount = (1<<ParticleCountPower);
	fprintf(stderr,"memory for CellIndex and CellParticle %d\n", PowerParticleCount );
	CellIndex    = (int *)malloc( (PowerParticleCount) * sizeof(int) );
    CellParticle = (int *)malloc( (PowerParticleCount) * sizeof(int) );
	#pragma acc enter data create(CellIndex[0:PowerParticleCount]) attach(CellIndex)
	#pragma acc enter data create(CellParticle[0:PowerParticleCount]) attach(CellParticle)
	
	#pragma acc update device(CellCount[0:DIM],TotalCellCount)
	#pragma acc update device(DomainMin[0:DIM],DomainMax[0:DIM],DomainWidth[0:DIM])
	#pragma acc update device(ParticleCountPower,PowerParticleCount)
	#pragma acc update device(MaxRadius)
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
	
	#pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
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
		}
	}
	
	// Fill zeros in CellParticleBegin and CellParticleEnd
	int power = 0;
	const int N = 2*TotalCellCount;
	while( (N>>power) != 0 ){
		power++;
	}
	const int powerN = (1<<power);
	
	int * ptr = (int *)malloc( powerN * sizeof(int));
	#pragma acc enter data create(ptr[0:powerN])
	
	#pragma acc kernels present(ptr[0:powerN])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<powerN;++iRow){
		ptr[iRow]=0;
	}
	
	#pragma acc kernels present(ptr[0:powerN],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		ptr[iC]               =CellFluidParticleEnd[iC]-CellFluidParticleBegin[iC];
		ptr[iC+TotalCellCount]=CellWallParticleEnd[iC] -CellWallParticleBegin[iC];
	}
	
	// Convert ptr to cumulative sum
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (1<<iMain);	
		#pragma acc kernels present(ptr[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			ptr[iRow]+=ptr[iRow+dist];
		}
	}
	for(int iMain=0;iMain<power;++iMain){
		const int dist = (powerN>>(iMain+1));	
		#pragma acc kernels present(ptr[0:powerN])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<powerN;iRow+=(dist<<1)){
			ptr[iRow]-=ptr[iRow+dist];
			ptr[iRow+dist]+=ptr[iRow];
		}
	}
	
	#pragma acc kernels present(ptr[0:powerN],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iC=0;iC<TotalCellCount;++iC){
		if(iC==0){	CellFluidParticleBegin[iC]=0;	}
		else     { 	CellFluidParticleBegin[iC]=ptr[iC-1];	}
		CellFluidParticleEnd[iC]  =ptr[iC];
		CellWallParticleBegin[iC] =ptr[iC-1+TotalCellCount];
		CellWallParticleEnd[iC]   =ptr[iC+TotalCellCount];
	}
	
	free(ptr);
	#pragma acc exit data delete(ptr[0:powerN])
	
	#pragma acc kernels present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount])
	{
		FluidParticleBegin = CellFluidParticleBegin[0];
		FluidParticleEnd   = CellFluidParticleEnd[TotalCellCount-1];
		WallParticleBegin  = CellWallParticleBegin[0];
		WallParticleEnd    = CellWallParticleEnd[TotalCellCount-1];
	}
	#pragma acc update host(FluidParticleBegin,FluidParticleEnd,WallParticleBegin,WallParticleEnd)
//	fprintf(stderr,"line:%d, FluidParticleBegin=%d, FluidParticleEnd=%d, WallParticleBegin=%d, WallParticleEnd=%d\n",__LINE__,FluidParticleBegin, FluidParticleEnd, WallParticleBegin, WallParticleEnd);
	
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
	
	const int rangeX = (int)(ceil(MaxRadius/CellWidth));
	const int rangeY = (int)(ceil(MaxRadius/CellWidth));
	#ifdef TWO_DIMENSIONAL
	const int rangeZ = 0;
	#else // not TWO_DIMENSIONAL (three dimensional) 
	const int rangeZ = (int)(ceil(MaxRadius/CellWidth));
	#endif
	
	#define MAX_1D_NEIGHBOR_CELL_COUNT 3
	assert( 2*rangeX+1 <= MAX_1D_NEIGHBOR_CELL_COUNT );
	assert( 2*rangeY+1 <= MAX_1D_NEIGHBOR_CELL_COUNT );
	assert( 2*rangeZ+1 <= MAX_1D_NEIGHBOR_CELL_COUNT );
	
	#pragma acc kernels present(CellParticle[0:PowerParticleCount],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellWallParticleBegin[0:TotalCellCount],CellWallParticleEnd[0:TotalCellCount],Position[0:ParticleCount][0:DIM])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<ParticleCount;++iP){
		const int iCX=(CellIndex[iP]/(CellCount[1]*CellCount[2]))%TotalCellCount;
		const int iCY=(CellIndex[iP]/CellCount[2])%CellCount[1];
		const int iCZ=CellIndex[iP]%CellCount[2];
		// same as
		// const int iCX=((int)floor((q[iP][0]-DomainMin[0])/CellWidth))%CellCount[0];
		// const int iCY=((int)floor((q[iP][1]-DomainMin[1])/CellWidth))%CellCount[1];
		// const int iCZ=((int)floor((q[iP][2]-DomainMin[2])/CellWidth))%CellCount[2];
		
		int jCXs[MAX_1D_NEIGHBOR_CELL_COUNT];
		int jCYs[MAX_1D_NEIGHBOR_CELL_COUNT];
		int jCZs[MAX_1D_NEIGHBOR_CELL_COUNT];
		
		#pragma acc loop seq
		for(int jX=0;jX<2*rangeX+1;++jX){
			jCXs[jX]=((iCX-rangeX+jX)%CellCount[0]+CellCount[0])%CellCount[0];
		}
		#pragma acc loop seq
		for(int jY=0;jY<2*rangeY+1;++jY){
			jCYs[jY]=((iCY-rangeY+jY)%CellCount[1]+CellCount[1])%CellCount[1];
		}
		#pragma acc loop seq
		for(int jZ=0;jZ<2*rangeZ+1;++jZ){
			jCZs[jZ]=((iCZ-rangeZ+jZ)%CellCount[2]+CellCount[2])%CellCount[2];
		}
		const int bX = (2*rangeX)-(iCX+rangeX)%CellCount[0];
		const int jXmin= ( ( bX>0 )? bX:0 );
		const int bY = (2*rangeY)-(iCY+rangeY)%CellCount[1];
		const int jYmin= ( ( bY>0 )? bY:0 );
		const int bZ = (2*rangeZ)-(iCZ+rangeZ)%CellCount[2];
		const int jZmin= ( ( bZ>0 )? bZ:0 );
		
		
		#pragma acc loop seq
		for(int jX=jXmin;jX<jXmin+(2*rangeX+1);++jX){
			#pragma acc loop seq
			for(int jY=jYmin;jY<jYmin+(2*rangeY+1);++jY){
				#pragma acc loop seq
				for(int jZ=jZmin;jZ<jZmin+(2*rangeZ+1);++jZ){
					const int jCX = jCXs[ jX % (2*rangeX+1)];
					const int jCY = jCYs[ jY % (2*rangeY+1)];
					const int jCZ = jCZs[ jZ % (2*rangeZ+1)];
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
		for(int jX=jXmin;jX<jXmin+(2*rangeX+1);++jX){
			#pragma acc loop seq
			for(int jY=jYmin;jY<jYmin+(2*rangeY+1);++jY){
				#pragma acc loop seq
				for(int jZ=jZmin;jZ<jZmin+(2*rangeZ+1);++jZ){
					const int jCX = jCXs[ jX % (2*rangeX+1)];
					const int jCY = jCYs[ jY % (2*rangeY+1)];
					const int jCZ = jCZs[ jZ % (2*rangeZ+1)];
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
		//if(VolStrainP[iP]<0.0){Lambda[iP]=0.0;}
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
static int    *CsrPtrA;  // [ FluidCount * DIM + 1 ] NeighborFluidCount�̐ώZ��Ƃ��Čv�Z��
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
	{
		NonzeroCountA=CsrPtrA[N];
	}
	#pragma acc update host(NonzeroCountA)
    
	// calculate coeeficient matrix A and source vector B
	CsrCofA = (double *)malloc( NonzeroCountA * sizeof(double) );
	CsrIndA = (int *)malloc( NonzeroCountA * sizeof(int) );
	VectorB = (double *)malloc( N * sizeof(double) );
	#pragma acc enter data create(CsrCofA[0:NonzeroCountA]) attach(CsrCofA)
	#pragma acc enter data create(CsrIndA[0:NonzeroCountA]) attach(CsrIndA)
	#pragma acc enter data create(VectorB[0:N]) attach(VectorB)
    
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

static void freeMatrixA( void ){
	free(CsrCofA);
	free(CsrIndA);
	free(CsrPtrA);
	free(VectorB);
	#pragma acc exit data delete(CsrCofA,CsrIndA,CsrPtrA,VectorB)
}

static int    NonzeroCountC;
static double *CsrCofC; // [ FluidCount * DIM x NeighCount ]
static int    *CsrIndC; // [ FluidCount * DIM x NeighCount ]
static int    *CsrPtrC; // [ FluidCount * DIM + 1 ] NeighCount�̐ώZ��
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
	#pragma acc enter data create(CsrPtrC[0:powerN]) attach(CsrPtrC)
	
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
	{
		NonzeroCountC = CsrPtrC[N];
	}
	#pragma acc update host(NonzeroCountC)
	
	// calculate coefficient matrix C and source vector P
	CsrCofC = (double *)malloc( NonzeroCountC * sizeof(double) );
	CsrIndC = (int *)malloc( NonzeroCountC * sizeof(int) );
	VectorP = (double *)malloc( ParticleCount * sizeof(double) );
	#pragma acc enter data create(CsrCofC[0:NonzeroCountC]) attach(CsrCofC)
	#pragma acc enter data create(CsrIndC[0:NonzeroCountC]) attach(CsrIndC)
	#pragma acc enter data create(VectorP[0:ParticleCount]) attach(VectorP)
	
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


static void freeMatrixC( void ){
	free(CsrCofC);
	free(CsrIndC);
	free(CsrPtrC);
	free(VectorP);
	#pragma acc exit data delete(CsrPtrC,CsrIndC,CsrCofC,VectorP)
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
	
	// A = A + C��C^T
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
					// break;
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
}



static int MultiGridDepth;
#pragma acc declare create(MultiGridDepth)

static int (*MultiGridCellMin)[DIM]; // grid range for each grid layer
static int (*MultiGridCellMax)[DIM];
static int (*MultiGridCount)[DIM];
#pragma acc declare create(MultiGridCellMin,MultiGridCellMax,MultiGridCount)

static int  (*MultiGridOffset); //�����ɉ��ڂ�Grid���炪iPower�̊K�w��������
#pragma acc declare create(MultiGridOffset)

static int TotalTopGridCount;
static int TopGridCount[DIM];
#pragma acc declare create(TotalTopGridCount,TopGridCount)

#define GridId(iGX,iGY,iGZ,gridcount) \
((gridcount[1]*gridcount[2])*(iGX)+(gridcount[2])*(iGY)+(iGZ))


static double *InvDiagA;
#pragma acc declare create(InvDiagA)

static int     CsrNnzP2G;
static int    *CsrPtrP2G;
static int    *CsrIndP2G;
static double *CsrCofP2G;
#pragma acc declare create(CsrNnzP2G,CsrPtrP2G,CsrIndP2G,CsrCofP2G)

static int     CsrNnzG2P;
static int    *CsrPtrG2P;
static int    *CsrIndG2P;
static double *CsrCofG2P;
#pragma acc declare create(CsrNnzG2P,CsrPtrG2P,CsrIndG2P,CsrCofG2P)


#ifdef TWO_DIMENSIONAL
#define offsetscale(power)  (((1<<(2*(power)))-1)/((1<<2)-1)) 
#else //Not TWO_DIMENSIONAL (three dimensional)
#define offsetscale(power)  (((1<<(3*(power)))-1)/((1<<3)-1))
#endif

#ifdef TWO_DIMENSIONAL
#define POW_2_DIM (2*2)
#define POW_3_DIM (3*3)
#else
#define POW_2_DIM (2*2*2)
#define POW_3_DIM (3*3*3)
#endif


static int     OneGridSizeVec = DIM;
static int     AllGridSizeVecS;
static double  (*MultiGridVecS);
#pragma acc declare create(MultiGridVecS)

static int     AllGridSizeVecR;
static double  (*MultiGridVecR);
#pragma acc declare create(MultiGridVecR)

static int     AllGridSizeVecQ;
static double  (*MultiGridVecQ);
#pragma acc declare create(MultiGridVecQ)

static int     (*MultiGridCsrNnzA);
static int     AllGridSizeCsrPtrA;
static int     OneGridSizeCsrPtrA = DIM;
static int     (*MultiGridCsrPtrA);
static int     AllGridSizeCsrIndA;
static int     OneGridSizeCsrIndA = DIM * DIM * POW_3_DIM;
static int     (*MultiGridCsrIndA);
static double  (*MultiGridCsrCofA);
#pragma acc declare create(MultiGridCsrPtrA)
#pragma acc declare create(MultiGridCsrIndA,MultiGridCsrCofA)

static int     AllGridSizeInvDiagA;
static int     OneGridSizeInvDiagA = DIM;
static double  (*MultiGridInvDiagA);
#pragma acc declare create(MultiGridInvDiagA)

static int     (*MultiGridCsrNnzR);
static int     AllGridSizeCsrPtrR;
static int     OneGridSizeCsrPtrR = DIM;
static int     (*MultiGridCsrPtrR);
static int     AllGridSizeCsrIndR;
static int     OneGridSizeCsrIndR = DIM * POW_2_DIM;
static int     (*MultiGridCsrIndR);
static double  (*MultiGridCsrCofR);
#pragma acc declare create(MultiGridCsrPtrR)
#pragma acc declare create(MultiGridCsrIndR,MultiGridCsrCofR)

static int     (*MultiGridCsrNnzP);
static int     AllGridSizeCsrPtrP;
static int     OneGridSizeCsrPtrP = DIM * POW_2_DIM;
static int     (*MultiGridCsrPtrP);
static int     AllGridSizeCsrIndP;
static int     OneGridSizeCsrIndP = DIM * POW_2_DIM;
static int     (*MultiGridCsrIndP);
static double  (*MultiGridCsrCofP);
#pragma acc declare create(MultiGridCsrPtrP)
#pragma acc declare create(MultiGridCsrIndP,MultiGridCsrCofP)

static void calculateMultiGridDepth( void );
static void allocateMultiGrid( void );
static void calculateCsrP2G( void );
static void calculateCsrG2P( void );
static void calculateInvDiagA( void );
static void calculateMultiGridCsrR( void );
static void calculateMultiGridCsrP( void );
static void calculateMultiGridCsrA( void );
static void calculateMultiGridInvDiagA( void );

static void checkMultiGridMatrix( void );


static void calculateMultiGridMatrix( void ){
	
	calculateMultiGridDepth();
	allocateMultiGrid();
	calculateCsrP2G();
	calculateCsrG2P();
	calculateInvDiagA();
	calculateMultiGridCsrR();
	calculateMultiGridCsrP();
	calculateMultiGridCsrA();
	calculateMultiGridInvDiagA();
	
	// checkMultiGridMatrix();
}


static void calculateMultiGridDepth( void ){
	
	int iCXmax=0;
	int iCYmax=0;
	int iCZmax=0;
	int iCXmin=CellCount[0];
	int iCYmin=CellCount[1];
	int iCZmin=CellCount[2];
		
	#pragma acc kernels copy(iCXmax,iCYmax,iCZmax,iCXmin,iCYmin,iCZmin) present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellCount[0:DIM])
	#pragma acc loop reduction(max:iCXmax,iCYmax,iCZmax) reduction(min:iCXmin,iCYmin,iCZmin)
	#pragma omp parallel for reduction (max:iCXmax,iCYmax,iCZmax) reduction(min:iCXmin,iCYmin,iCZmin)
	for(int iC=0;iC<TotalCellCount;++iC){
		if(CellFluidParticleEnd[iC]-CellFluidParticleBegin[iC]>0){
			const int iCX=(iC/(CellCount[1]*CellCount[2]))%CellCount[0];
			const int iCY=(iC/CellCount[2])%CellCount[1];
			const int iCZ=iC%CellCount[2];
			
			if(iCXmax<iCX){iCXmax=iCX;}
			if(iCYmax<iCY){iCYmax=iCY;}
			if(iCZmax<iCZ){iCZmax=iCZ;}
			if(iCXmin>iCX){iCXmin=iCX;}
			if(iCYmin>iCY){iCYmin=iCY;}
			if(iCZmin>iCZ){iCZmin=iCZ;}
		}
	}
	
	double maxGridWidth=0.0;
	if(CellWidth*(iCXmax-iCXmin)>maxGridWidth){
		maxGridWidth=CellWidth*(iCXmax-iCXmin+1);
	}
	if(CellWidth*(iCYmax-iCYmin)>maxGridWidth){
		maxGridWidth=CellWidth*(iCYmax-iCYmin+1);
	}
	if(CellWidth*(iCZmax-iCZmin)>maxGridWidth){
		maxGridWidth=CellWidth*(iCZmax-iCZmin+1);
	}
	
	{
		int iPower=0;
		while((1<<iPower)*CellWidth <= maxGridWidth){
			iPower++;
		}
		MultiGridDepth=iPower;
	}
//	#pragma acc update device(MultiGridDepth)
	
	MultiGridCellMin = (int (*)[DIM])malloc( MultiGridDepth*sizeof(int [DIM]) );
	MultiGridCellMax = (int (*)[DIM])malloc( MultiGridDepth*sizeof(int [DIM]) );
	MultiGridCount   = (int (*)[DIM])malloc( MultiGridDepth*sizeof(int [DIM]) );
	#pragma acc enter data create(MultiGridCellMin[0:MultiGridDepth][0:DIM],MultiGridCellMax[0:MultiGridDepth][0:DIM],MultiGridCount[0:MultiGridDepth][0:DIM])
	
	for(int iPower=0;iPower<MultiGridDepth;++iPower){
		MultiGridCellMin[iPower][0] = (iCXmin>>iPower);
		MultiGridCellMin[iPower][1] = (iCYmin>>iPower);
		MultiGridCellMin[iPower][2] = (iCZmin>>iPower);
	}
	for(int iPower=0;iPower<MultiGridDepth;++iPower){
		MultiGridCellMax[iPower][0] = (iCXmax>>iPower)+1;
		MultiGridCellMax[iPower][1] = (iCYmax>>iPower)+1;
		MultiGridCellMax[iPower][2] = (iCZmax>>iPower)+1;
	}
	for(int iPower=0;iPower<MultiGridDepth;++iPower){
		MultiGridCount[iPower][0] = MultiGridCellMax[iPower][0] - MultiGridCellMin[iPower][0];
		MultiGridCount[iPower][1] = MultiGridCellMax[iPower][1] - MultiGridCellMin[iPower][1];
		MultiGridCount[iPower][2] = MultiGridCellMax[iPower][2] - MultiGridCellMin[iPower][2];
	}
	#pragma acc update device(MultiGridCellMin[0:MultiGridDepth][0:DIM],MultiGridCellMax[0:MultiGridDepth][0:DIM],MultiGridCount[0:MultiGridDepth][0:DIM])
	
	MultiGridOffset = (int *)malloc( (MultiGridDepth+1)*sizeof(int) );
	MultiGridOffset[MultiGridDepth-MultiGridDepth]=0;
	for(int iPower=MultiGridDepth-1;iPower>=0;--iPower){
		MultiGridOffset[MultiGridDepth-(iPower-1)-1] = MultiGridOffset[MultiGridDepth-iPower-1] + MultiGridCount[iPower][0]*MultiGridCount[iPower][1]*MultiGridCount[iPower][2];
	}

//	log_printf("line:%d, MultiGridDepth=%d\n", __LINE__, MultiGridDepth);
//	for(int iPower=0;iPower<MultiGridDepth;++iPower){
//		log_printf("MultiGridCellMin[%d]=(%d,%d,%d)\n",iPower, MultiGridCellMin[iPower][0],MultiGridCellMin[iPower][1],MultiGridCellMin[iPower][2]);
//		log_printf("MultiGridCellMax[%d]=(%d,%d,%d)\n",iPower, MultiGridCellMax[iPower][0],MultiGridCellMax[iPower][1],MultiGridCellMax[iPower][2]);
//		log_printf("MultiGridCount[%d]  =(%d,%d,%d)\n",iPower, MultiGridCount[iPower][0],  MultiGridCount[iPower][1],  MultiGridCount[iPower][2]);
//	}
	
}



static void allocateMultiGrid( void ){ // MultiGridParticleCount, MultiInvD, MultiVecS, MultiVecR, MultiVecQ
	
	const int fluidcount=FluidParticleEnd-FluidParticleBegin;
	const int totalGridCount = MultiGridCount[0][0]*MultiGridCount[0][1]*MultiGridCount[0][2];
	
	InvDiagA = (double *)malloc( DIM*fluidcount * sizeof(double) );
	#pragma acc enter data create( InvDiagA[0:DIM*fluidcount] ) //attach(InvDiagA)
	
	CsrNnzP2G = DIM*fluidcount;
	CsrPtrP2G = (int *)malloc( (DIM*totalGridCount+1) * sizeof(int) );
	#pragma acc enter data create( CsrPtrP2G[0:DIM*totalGridCount+1] ) //attach(CsrPtrP2G)
	CsrIndP2G = (int *)malloc( DIM*fluidcount * sizeof(int) );
	#pragma acc enter data create( CsrIndP2G[0:DIM*fluidcount] ) //attach(CsrIndP2G)
	CsrCofP2G = (double *)malloc( DIM*fluidcount * sizeof(double) );
	#pragma acc enter data create( CsrCofP2G[0:DIM*fluidcount] ) //attach(CsrCofP2G)
	
	CsrNnzG2P = DIM*fluidcount;
	CsrPtrG2P = (int *)malloc( (DIM*fluidcount+1) * sizeof(int) );
	#pragma acc enter data create( CsrPtrG2P[0:DIM*fluidcount+1] ) //attach(CsrPtrG2P)
	CsrIndG2P = (int *)malloc( DIM*fluidcount * sizeof(int) );
	#pragma acc enter data create( CsrIndG2P[0:DIM*fluidcount] ) //attach(CsrIndG2P)
	CsrCofG2P = (double *)malloc( DIM*fluidcount * sizeof(double) );
	#pragma acc enter data create( CsrCofG2P[0:DIM*fluidcount] ) //attach(CsrCofG2P)
	
	AllGridSizeVecS = MultiGridOffset[MultiGridDepth]*OneGridSizeVec;
	MultiGridVecS   = (double *)malloc( AllGridSizeVecS * sizeof(double) );
	#pragma acc enter data create(MultiGridVecS[0:AllGridSizeVecS]) //attach(MultiGridVecS)
	
	AllGridSizeVecR = MultiGridOffset[MultiGridDepth]*OneGridSizeVec;
	MultiGridVecR   = (double *)malloc( AllGridSizeVecR * sizeof(double) );
	#pragma acc enter data create(MultiGridVecR[0:AllGridSizeVecR]) //attach(MultiGridVecR)
	
	AllGridSizeVecQ = MultiGridOffset[MultiGridDepth]*OneGridSizeVec;
	MultiGridVecQ   = (double *)malloc( AllGridSizeVecQ * sizeof(double) );
	#pragma acc enter data create(MultiGridVecQ[0:AllGridSizeVecQ]) //attach(MultiGridVecQ)
	
	
	MultiGridCsrNnzA = (int *)malloc( MultiGridDepth * sizeof(int) );
	AllGridSizeCsrPtrA = MultiGridOffset[MultiGridDepth]*OneGridSizeCsrPtrA + (MultiGridDepth);
	MultiGridCsrPtrA   = (int *)malloc( AllGridSizeCsrPtrA * sizeof(int) );
	#pragma acc enter data create(MultiGridCsrPtrA[0:AllGridSizeCsrPtrA]) //attach(MultiGridCsrPtrA)
	AllGridSizeCsrIndA = MultiGridOffset[MultiGridDepth]*OneGridSizeCsrIndA;
	MultiGridCsrIndA   = (int *)malloc( AllGridSizeCsrIndA * sizeof(int) );
	#pragma acc enter data create(MultiGridCsrIndA[0:AllGridSizeCsrIndA]) //attach(MultiGridCsrIndA)
	MultiGridCsrCofA   = (double *)malloc( AllGridSizeCsrIndA * sizeof(double) );
	#pragma acc enter data create(MultiGridCsrCofA[0:AllGridSizeCsrIndA]) //attach(MultiGridCsrCofA)
	
	AllGridSizeInvDiagA = MultiGridOffset[MultiGridDepth]*OneGridSizeInvDiagA;
	MultiGridInvDiagA   = (double *)malloc( AllGridSizeInvDiagA * sizeof(double) );
	#pragma acc enter data create(MultiGridInvDiagA[0:AllGridSizeInvDiagA]) //attach(MultiGridInvDiagA)
	
	MultiGridCsrNnzR = (int *)malloc( MultiGridDepth * sizeof(int) );
	AllGridSizeCsrPtrR = MultiGridOffset[MultiGridDepth-1]*OneGridSizeCsrPtrR + (MultiGridDepth-1);
	MultiGridCsrPtrR   = (int *)malloc( AllGridSizeCsrPtrR * sizeof(int) );
	#pragma acc enter data create(MultiGridCsrPtrR[0:AllGridSizeCsrPtrR])// attach(MultiGridCsrPtrR)
	AllGridSizeCsrIndR = MultiGridOffset[MultiGridDepth-1]*OneGridSizeCsrIndR;
	MultiGridCsrIndR   = (int *)malloc( AllGridSizeCsrIndR * sizeof(int) );
	#pragma acc enter data create(MultiGridCsrIndR[0:AllGridSizeCsrIndR])// attach(MultiGridCsrIndR)
	MultiGridCsrCofR   = (double *)malloc( AllGridSizeCsrIndR * sizeof(double) );
	#pragma acc enter data create(MultiGridCsrCofR[0:AllGridSizeCsrIndR])// attach(MultiGridCsrCofR)
	
	MultiGridCsrNnzP = (int *)malloc( MultiGridDepth * sizeof(int) );
	AllGridSizeCsrPtrP = MultiGridOffset[MultiGridDepth-1]*OneGridSizeCsrPtrP + (MultiGridDepth-1);
	MultiGridCsrPtrP   = (int *)malloc( AllGridSizeCsrPtrP * sizeof(int) );
	#pragma acc enter data create(MultiGridCsrPtrP[0:AllGridSizeCsrPtrP]) //attach(MultiGridCsrPtrP)
	AllGridSizeCsrIndP = MultiGridOffset[MultiGridDepth-1]*OneGridSizeCsrIndP;
	MultiGridCsrIndP   = (int *)malloc( AllGridSizeCsrIndP * sizeof(int) );
	#pragma acc enter data create(MultiGridCsrIndP[0:AllGridSizeCsrIndP]) //attach(MultiGridCsrIndP)
	MultiGridCsrCofP   = (double *)malloc( AllGridSizeCsrIndP * sizeof(double) );
	#pragma acc enter data create(MultiGridCsrCofP[0:AllGridSizeCsrIndP]) //attach(MultiGridCsrCofP)
	
}



static void calculateCsrP2G( void ){
	
	const int fluidcount=FluidParticleEnd-FluidParticleBegin;
	const int iPower=0;
	const int *gridCellMin = MultiGridCellMin[iPower];
	const int *gridCellMax = MultiGridCellMax[iPower];
	const int *gridCount   = MultiGridCount[iPower];
	const int totalGridCount = gridCount[0]*gridCount[1]*gridCount[2];
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<DIM*totalGridCount+1;++iRow){
		CsrPtrP2G[iRow]=-1*__LINE__;
	}
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iNonzero=0;iNonzero<CsrNnzP2G;++iNonzero){
		CsrIndP2G[iNonzero]=-1*__LINE__;
		CsrCofP2G[iNonzero]=-1.0*__LINE__;
	}
	
	#pragma acc kernels present(gridCount[0:DIM],gridCellMin[0:DIM]) present(CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CsrPtrP2G[0:DIM*totalGridCount+1],CsrIndP2G[0:DIM*fluidcount], CsrCofP2G[0:DIM*fluidcount])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iG=0;iG<totalGridCount;++iG){
		const int iGX = (iG/(gridCount[1]*gridCount[2])) % gridCount[0];
		const int iGY = (iG/(gridCount[2])) % gridCount[1];
		const int iGZ =  iG % gridCount[2];
		const int iCX = iGX + gridCellMin[0];
		const int iCY = iGY + gridCellMin[1];
		const int iCZ = iGZ + gridCellMin[2];
		
		const int iC = CellId(iCX,iCY,iCZ);
		const int ptr = CellFluidParticleBegin[iC];
		const int count = CellFluidParticleEnd[iC]-CellFluidParticleBegin[iC];
		
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			CsrPtrP2G[DIM*iG+rD] = ptr*DIM+count*rD;
		}
		if(iG==totalGridCount-1){
			CsrPtrP2G[DIM*totalGridCount]=CellFluidParticleEnd[iC]*DIM;
		}
		
		if( count==0 )continue;
		#pragma acc loop seq
		for(int jP=CellFluidParticleBegin[iC];jP<CellFluidParticleEnd[iC];++jP){
			#pragma acc loop seq
			for(int rD=0;rD<DIM;++rD){
				const int iRow=DIM*iG+rD;
				const int jColumn=DIM*jP+rD;
				const int jCP = jP-CellFluidParticleBegin[iC];
				const int iNonzero=CsrPtrP2G[iRow]+jCP;
				CsrIndP2G[iNonzero]=jColumn;
				CsrCofP2G[iNonzero]=1.0;
			}
		}
	}
}


static void calculateCsrG2P( void ){
	
	const int fluidcount=FluidParticleEnd-FluidParticleBegin;
	const int iPower=0;
	const int *gridCellMin = MultiGridCellMin[iPower];
	const int *gridCellMax = MultiGridCellMax[iPower];
	const int *gridCount   = MultiGridCount[iPower];
	const int totalGridCount = gridCount[0]*gridCount[1]*gridCount[2];
	
	#pragma acc kernels present(gridCount[0:DIM],gridCellMin[0:DIM]) present(CellCount[0:DIM],CsrPtrG2P[0:DIM*fluidcount+1],CsrIndG2P[0:CsrNnzG2P],CsrCofG2P[0:CsrNnzG2P])
	#pragma acc loop independent 
	#pragma omp parallel for 
	for(int iP=0;iP<fluidcount;++iP){
		const int iC=CellIndex[iP];
		const int iCX = iC/(CellCount[1]*CellCount[2])%CellCount[0];
		const int iCY = iC/(CellCount[2])%CellCount[1];
		const int iCZ = iC%CellCount[2];
		const int iGX = iCX-gridCellMin[0];
		const int iGY = iCY-gridCellMin[1];
		const int iGZ = iCZ-gridCellMin[2];
		const int iG=GridId(iGX,iGY,iGZ,gridCount);
		const int count = CellFluidParticleEnd[iC]-CellFluidParticleBegin[iC];
		
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow = DIM*iP+rD;
			CsrPtrG2P[iRow]=DIM*iP+rD;
		}
		if(iP==fluidcount-1){
			const int iRow = DIM*fluidcount;
			CsrPtrG2P[iRow] = DIM*fluidcount;
			//assert( CsrPtrG2P[iRow]==GsrNnzG2P );
		}
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow=DIM*iP+rD;
			const int iColumn=DIM*iG+rD;
			const int iNonzero=CsrPtrG2P[iRow]+0;
			CsrIndG2P[iNonzero]=iColumn;
			CsrCofG2P[iNonzero]=1.0;
		}
	}
}

static void calculateInvDiagA( void ){
	const int fluidcount=FluidParticleEnd-FluidParticleBegin;
	const int N = DIM*(fluidcount);
	
	#pragma acc kernels
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<N;++iRow){
		InvDiagA[iRow] = -1.0*__LINE__;;
	}
	
	#pragma acc kernels present(InvDiagA[0:N], CsrPtrA[0:N+1],CsrIndA[0:NonzeroCountA],CsrCofA[0:NonzeroCountA])
	#pragma acc loop independent
	#pragma omp parallel for 
	for(int iRow=0;iRow<N;++iRow){
		double sumCof=0.0;
		#pragma acc loop seq
		for(int iNonzero=CsrPtrA[iRow];iNonzero<CsrPtrA[iRow+1];++iNonzero){
			sumCof+=abs(CsrCofA[iNonzero]);
		}
		if(sumCof!=0.0){
			InvDiagA[iRow] = 2.0/sumCof;
		}
		else{
			InvDiagA[iRow] = 0.0;
		}
	}
	
}

static void calculateMultiGridCsrR( void ){
	
	
	for(int iPower=1;iPower<MultiGridDepth;++iPower){
		const int *gridCount =MultiGridCount[iPower-1];
		MultiGridCsrNnzR[iPower] = DIM*gridCount[0]*gridCount[1]*gridCount[2];
	}
	
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<AllGridSizeCsrPtrR;++iRow){
		MultiGridCsrPtrR[iRow]=-1*__LINE__;
	}
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iNonzero=0;iNonzero<AllGridSizeCsrIndR;++iNonzero){
		MultiGridCsrIndR[iNonzero]=-1*__LINE__;
		MultiGridCsrCofR[iNonzero]=-1.0*__LINE__;
	}
	
	for(int iPower=1;iPower<MultiGridDepth;++iPower){
		const int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrR + (MultiGridDepth-iPower-1) ;
		const int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndR;
		const int *gridCellMinL     = MultiGridCellMin[iPower];
		const int *gridCellMaxL     = MultiGridCellMax[iPower];
		const int *gridCountL       = MultiGridCount[iPower];
		const int *gridCellMinS     = MultiGridCellMin[iPower-1];
		const int *gridCellMaxS     = MultiGridCellMax[iPower-1];
		const int *gridCountS       = MultiGridCount[iPower-1];
		const int NRow = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
		const int nnz = MultiGridCsrNnzR[iPower];
		int    *csrPtrR = &MultiGridCsrPtrR[offsetPtr];
		int    *csrIndR = &MultiGridCsrIndR[offsetInd];
		double *csrCofR = &MultiGridCsrCofR[offsetInd];
		
		#pragma acc kernels present(                                   \
			gridCellMinL[0:DIM],gridCellMaxL[0:DIM],gridCountL[0:DIM], \
			gridCellMinS[0:DIM],gridCellMaxS[0:DIM],gridCountS[0:DIM], \
			csrPtrR[0:NRow+1],csrIndR[0:nnz],csrCofR[0:nnz]            \
		)
		#pragma acc loop independent
		#pragma omp parallel for 
		for(int iRow=0;iRow<NRow;++iRow){
			const int iLGX = (iRow/(DIM*gridCountL[1]*gridCountL[2])) % gridCountL[0];
			const int iLGY = (iRow/(DIM*gridCountL[2])) % gridCountL[1];
			const int iLGZ = (iRow/DIM) % gridCountL[2];
			const int rD   =  iRow % DIM;
			
			const int gmX = ( (gridCellMinS[0]>2*gridCellMinL[0]) ? 1:0 );
			const int gmY = ( (gridCellMinS[1]>2*gridCellMinL[1]) ? 1:0 );
			const int gmZ = ( (gridCellMinS[2]>2*gridCellMinL[2]) ? 1:0 );
			const int gMX = ( (gridCellMaxS[0]<2*gridCellMaxL[0]) ? 1:0 );
			const int gMY = ( (gridCellMaxS[1]<2*gridCellMaxL[1]) ? 1:0 );
			const int gMZ = ( (gridCellMaxS[2]<2*gridCellMaxL[2]) ? 1:0 );
			const int iLCX = iLGX + gridCellMinL[0];
			const int iLCY = iLGY + gridCellMinL[1];
			const int iLCZ = iLGZ + gridCellMinL[2];
			const int fmX = ( (gridCellMinS[0]>2*iLCX) ? 1:0 );
			const int fmY = ( (gridCellMinS[1]>2*iLCY) ? 1:0 );
			const int fmZ = ( (gridCellMinS[2]>2*iLCZ) ? 1:0 );
			const int fMX = ( (gridCellMaxS[0]<2*(iLCX+1)) ? 1:0 );
			const int fMY = ( (gridCellMaxS[1]<2*(iLCY+1)) ? 1:0 );
			const int fMZ = ( (gridCellMaxS[2]<2*(iLCZ+1)) ? 1:0 );
			const int wX = gridCountS[0]; 
			const int wY = gridCountS[1];
			const int wZ = gridCountS[2];
			const int sX = 2*iLGX-gmX+fmX;
			const int sY = 2*iLGY-gmY+fmY;
			const int sZ = 2*iLGZ-gmZ+fmZ;
			const int cX = 2-fmX-fMX;
			const int cY = 2-fmY-fMY;
			const int cZ = 2-fmZ-fMZ;
			
			csrPtrR[ iRow ] = DIM*(wZ*wY*sX+wZ*sY*cX+sZ*cY*cX)+rD*cZ*cY*cX;
			
			if(iRow==NRow-1){
				const int iRow_end = NRow;
				csrPtrR[ iRow_end ] = DIM*wZ*wY*wX;
				// assert(MultiGridCsrPtrR[ offsetPtr +iRow_end ]==MultiGridCsrNnzR[iPower]);
			}
			
			#pragma acc loop seq
			for(int iX=0;iX<cX;++iX)
			#pragma acc loop seq
			for(int iY=0;iY<cY;++iY)
			#pragma acc loop seq
			for(int iZ=0;iZ<cZ;++iZ)
			{
				const int dX = iX+fmX;
				const int dY = iY+fmY;
				const int dZ = iZ+fmZ;
				const int iSGX=(2*iLCX+dX)-gridCellMinS[0];
				const int iSGY=(2*iLCY+dY)-gridCellMinS[1];
				const int iSGZ=(2*iLCZ+dZ)-gridCellMinS[2];;
				const int iSG=GridId(iSGX,iSGY,iSGZ,gridCountS);
				const int shift = cZ*cY*(dX-fmX)+cZ*(dY-fmY)+(dZ-fmZ);
				
				const int iColumn = DIM*iSG+rD;
				const int iNonzero = csrPtrR[ iRow ] + shift;
				csrIndR[ iNonzero ] = iColumn;
				csrCofR[ iNonzero ] = 1.0;
			}
		}
	}
}

static void calculateMultiGridCsrP( void ){
	
	for(int iPower=1;iPower<MultiGridDepth;++iPower){
		const int *gridCount =MultiGridCount[iPower-1];
		MultiGridCsrNnzP[iPower] = DIM*gridCount[0]*gridCount[1]*gridCount[2];
	}
	
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for 
	for(int iRow=0;iRow<AllGridSizeCsrPtrP;++iRow){
		MultiGridCsrPtrP[iRow]=-1*__LINE__;
	}
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for 
	for(int iNonzero=0;iNonzero<AllGridSizeCsrIndP;++iNonzero){
		MultiGridCsrIndP[iNonzero]=-1*__LINE__;
		MultiGridCsrCofP[iNonzero]=-1.0*__LINE__;
	}
	
	for(int iPower=MultiGridDepth-1;iPower>0;iPower--){
		const int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrP + (MultiGridDepth-iPower-1);
		const int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndP;
		const int *gridCellMinL     = MultiGridCellMin[iPower];
		const int *gridCellMinS     = MultiGridCellMin[iPower-1];
		const int *gridCountL       = MultiGridCount[iPower];
		const int *gridCountS       = MultiGridCount[iPower-1];
		const int NRow = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
		const int nnz = MultiGridCsrNnzP[iPower];
		int    *csrPtrP = &MultiGridCsrPtrP[offsetPtr];
		int    *csrIndP = &MultiGridCsrIndP[offsetInd];
		double *csrCofP = &MultiGridCsrCofP[offsetInd];
		
		#pragma acc kernels present(gridCellMinL[0:DIM],gridCellMinS[0:DIM],gridCountL[0:DIM],gridCountS[0:DIM]) present(csrPtrP[0:NRow+1],csrIndP[0:nnz],csrCofP[0:nnz])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<NRow;++iRow){
			const int iSGX = (iRow/(DIM*gridCountS[1]*gridCountS[2])) % gridCountS[0];
			const int iSGY = (iRow/(DIM*gridCountS[2])) % gridCountS[1];
			const int iSGZ = (iRow/DIM) % gridCountS[2];
			const int rD   =  iRow % DIM;
			csrPtrP[ iRow ] = iRow;
			
			if( iRow==NRow-1 ){
				const int iRow_end = NRow;
				csrPtrP[ iRow_end ] = iRow_end;
				// assert(MultiGridCsrPtrP[ offsetPtr +iRow_end ]==MultiGridCsrNnzP[iPower]);
			}
			const int iSCX = iSGX+gridCellMinS[0];
			const int iSCY = iSGY+gridCellMinS[1];
			const int iSCZ = iSGZ+gridCellMinS[2];
			
			const int iLGX = (iSCX/2)-gridCellMinL[0];
			const int iLGY = (iSCY/2)-gridCellMinL[1];
			const int iLGZ = (iSCZ/2)-gridCellMinL[2];
			const int iLG = GridId(iLGX,iLGY,iLGZ,gridCountL);
			const int iColumn = DIM*iLG+rD;
			const int iNonzero = csrPtrP[ iRow ] +0;
			csrIndP[ iNonzero ]=iColumn;
			csrCofP[ iNonzero ]=1.0;
			
		}
	}
}

static void calculateMultiGridCsrA( void ){
	const int dimension = 2;
	
	for(int iPower=0;iPower<MultiGridDepth;++iPower){
		const int *gridCount = MultiGridCount[iPower];
		const int cX  = ( 3<gridCount[0] ? 3:gridCount[0] );
		const int cY  = ( 3<gridCount[1] ? 3:gridCount[1] );
		const int cZ  = ( 3<gridCount[2] ? 3:gridCount[2] );
		const int wX = cX*gridCount[0];
		const int wY = cY*gridCount[1];
		const int wZ = cZ*gridCount[2];
		MultiGridCsrNnzA[iPower]=DIM*DIM * wX*wY*wZ;
	}
	
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for 
	for(int iRow=0;iRow<AllGridSizeCsrPtrA;++iRow){
		MultiGridCsrPtrA[iRow]=-1*__LINE__;
	}
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for 
	for(int iNonzero=0;iNonzero<AllGridSizeCsrIndA;++iNonzero){
		MultiGridCsrIndA[iNonzero]=-1*__LINE__;
		MultiGridCsrCofA[iNonzero]=-1.0*__LINE__;
	}
	
	
	for(int iPower=0;iPower<MultiGridDepth;++iPower){
		const int offsetPtrL = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
		const int offsetIndL = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
		const int *gridCountL       = MultiGridCount[iPower];
		const int NL = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
		const int nnzL = MultiGridCsrNnzA[iPower];
		int    *csrPtrAL = &MultiGridCsrPtrA[offsetPtrL];
		int    *csrIndAL=&MultiGridCsrIndA[offsetIndL];
		double *csrCofAL=&MultiGridCsrCofA[offsetIndL];
		
		#pragma acc kernels present(gridCountL[0:DIM]) present(csrPtrAL[0:NL+1],csrIndAL[0:nnzL],csrCofAL[0:nnzL])
		#pragma acc loop independent //collapse(4)
		#pragma omp parallel for //collapse(4)
		for(int iRowL=0;iRowL<NL;++iRowL){
			const int iLGX = (iRowL/(DIM*gridCountL[1]*gridCountL[2])) % gridCountL[0];
			const int iLGY = (iRowL/(DIM*gridCountL[2])) % gridCountL[1];
			const int iLGZ = (iRowL/DIM) % gridCountL[2];
			const int rD   =  iRowL % DIM;
			
			const int cX  = ( 3<gridCountL[0] ? 3:gridCountL[0] );
			const int cY  = ( 3<gridCountL[1] ? 3:gridCountL[1] );
			const int cZ  = ( 3<gridCountL[2] ? 3:gridCountL[2] );
			const int wX  = cX*gridCountL[0];
			const int wY  = cY*gridCountL[1];
			const int wZ  = cZ*gridCountL[2];
			const int sX  = cX*iLGX;
			const int sY  = cY*iLGY;
			const int sZ  = cZ*iLGZ;
			
			csrPtrAL[ iRowL ] = DIM*(DIM*( wZ*wY*sX + wZ*sY*cX + sZ*cY*cX ) + rD*( cZ*cY*cX )); // DIM*POW_3_DIM*iRowL;
			
			if(iRowL==NL-1){ 
				const int iRowL_end = NL;
				csrPtrAL[ iRowL_end ] = DIM*DIM * wZ*wY*wX;
				// assert(MultiGridCsrPtrA[ offsetPtrL +iRowL_end ]==MultiGridCsrNnzA[iPower]);
			}
			#pragma acc loop seq
			for(int iX=0;iX<cX;++iX)
			#pragma acc loop seq
			for(int iY=0;iY<cY;++iY)
			#pragma acc loop seq
			for(int iZ=0;iZ<cZ;++iZ){
				#pragma acc loop seq
				for(int sD=0;sD<DIM;++sD){
					const int shift = DIM*( cZ*cY*iX + cZ*iY + iZ ) +sD; 
					const int iNonzeroL = csrPtrAL[ iRowL ] + shift;
					const int dLGX = iX-1;
					const int dLGY = iY-1;
					const int dLGZ = iZ-1;
					int jLGX = (iLGX + dLGX + gridCountL[0])%gridCountL[0];
					int jLGY = (iLGY + dLGY + gridCountL[1])%gridCountL[1];
					int jLGZ = (iLGZ + dLGZ + gridCountL[2])%gridCountL[2];
					const int jLG=GridId(jLGX,jLGY,jLGZ,gridCountL);
					// assert( shift == DIM*(cZ*cY*(dLGX+1-fmX)+cZ*(dLGY+1-fmY)+(dLGZ+1-fmZ))+sD );
					const int iColumnL = DIM*jLG+sD;
					csrIndAL[ iNonzeroL ] = iColumnL;
					csrCofAL[ iNonzeroL ] = 1.0*__LINE__;
				}
			}
		}
	}
	
	{
		const int iPower=0;
		const int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
		const int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
		const int *gridCellMin = MultiGridCellMin[iPower];
		const int *gridCount = MultiGridCount[iPower];
		const int NG = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		const int NP = DIM*(FluidParticleEnd-FluidParticleBegin);
		const int nnzG = MultiGridCsrNnzA[iPower];
		int    *csrPtrAG = &MultiGridCsrPtrA[offsetPtr];
		double *csrCofAG = &MultiGridCsrCofA[offsetInd];
		
		#pragma acc kernels present(gridCellMin[0:DIM],gridCount[0:DIM],csrPtrAG[0:NG+1],csrCofAG[0:nnzG], CsrPtrA[0:NP+1],CsrIndA[0:NonzeroCountA],CsrCofA[0:NonzeroCountA],CellFluidParticleBegin[0:TotalCellCount],CellFluidParticleEnd[0:TotalCellCount],CellIndex[0:ParticleCount],CellCount[0:DIM],gridCellMin[0:DIM])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRowG=0;iRowG<NG;++iRowG){
			const int iGX = (iRowG/(DIM*gridCount[1]*gridCount[2])) % gridCount[0];
			const int iGY = (iRowG/(DIM*gridCount[2])) % gridCount[1];
			const int iGZ = (iRowG/DIM) % gridCount[2];
			const int rD  = iRowG % DIM;
			
			const int cX  = ( 3<gridCount[0] ? 3:gridCount[0] );
			const int cY  = ( 3<gridCount[1] ? 3:gridCount[1] );
			const int cZ  = ( 3<gridCount[2] ? 3:gridCount[2] );
			
			double cofRowG[DIM*POW_3_DIM];
			#pragma acc loop seq
			for(int shift=0;shift<DIM*POW_3_DIM;++shift){
				cofRowG[shift]=0.0;
			}
			
			const int iCX = iGX + gridCellMin[0];
			const int iCY = iGY + gridCellMin[1];
			const int iCZ = iGZ + gridCellMin[2];
			if(!(iCX<CellCount[0] && iCY<CellCount[1] && iCZ<CellCount[2]))continue;
			const int iC=CellId(iCX,iCY,iCZ);
			
			#pragma acc loop seq
			for(int iP=CellFluidParticleBegin[iC];iP<CellFluidParticleEnd[iC];++iP){
				const int iRowP=DIM*iP+rD;				
				#pragma acc loop seq
				for(int iNonzeroP=CsrPtrA[iRowP];iNonzeroP<CsrPtrA[iRowP+1];++iNonzeroP){
					const int iColumnP =CsrIndA[iNonzeroP];
					const int jP = iColumnP/DIM;
					const int sD = iColumnP%DIM;
					const int jCX=(CellIndex[jP]/(CellCount[1]*CellCount[2]))%TotalCellCount;
					const int jCY=(CellIndex[jP]/CellCount[2])%CellCount[1];
					const int jCZ=CellIndex[jP]%CellCount[2];
					const int jGX = jCX - gridCellMin[0];
					const int jGY = jCY - gridCellMin[1];
					const int jGZ = jCZ - gridCellMin[2];
					const int dGX = jGX-iGX;
					const int dGY = jGY-iGY;
					const int dGZ = jGZ-iGZ;
					const int iX = (dGX+1+gridCount[0])%gridCount[0];
					const int iY = (dGY+1+gridCount[1])%gridCount[1];
					const int iZ = (dGZ+1+gridCount[2])%gridCount[2];
					const int shift = DIM*(cZ*cY*iX+cZ*iY+iZ)+sD;
					
					#ifndef _OPENACC
					 const int iNonzeroG = csrPtrAG[ iRowG ] + shift;
					 const int jG  = GridId(jGX,jGY,jGZ,gridCount);
					 const int iColumnG = DIM*jG+sD;
					 assert( MultiGridCsrIndA[ offsetInd + iNonzeroG ] == iColumnG );
					#endif
					
					cofRowG[shift] += CsrCofA[iNonzeroP];
				}
			}
			#pragma acc loop seq
			for(int shift=0;shift<(DIM*cZ*cY*cX);++shift){
				const int iNonzeroG = csrPtrAG[ iRowG ] + shift;
				csrCofAG[ iNonzeroG ] = cofRowG[shift];
			}
		}
	}
	
	
	for(int iPower=1;iPower<MultiGridDepth;++iPower){
		const int offsetPtrL = MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrA + (MultiGridDepth-(iPower  )-1);
		const int offsetPtrS = MultiGridOffset[MultiGridDepth-(iPower-1)-1]*OneGridSizeCsrPtrA + (MultiGridDepth-(iPower-1)-1);
		const int offsetIndL = MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndA;
		const int offsetIndS = MultiGridOffset[MultiGridDepth-(iPower-1)-1]*OneGridSizeCsrIndA;
		const int offsetPtrR = MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrR + (MultiGridDepth-(iPower  )-1);
		const int offsetIndR = MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndR;
		const int *gridCellMinL     = MultiGridCellMin[iPower];
		const int *gridCellMinS     = MultiGridCellMin[iPower-1];
		const int *gridCountL = MultiGridCount[iPower];
		const int *gridCountS = MultiGridCount[iPower-1];
		const int NL = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
		const int NS = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
		const int nnzL = MultiGridCsrNnzA[iPower];
		const int nnzS = MultiGridCsrNnzA[iPower-1];		
		int    *csrPtrAL = &MultiGridCsrPtrA[offsetPtrL];
		int    *csrPtrAS = &MultiGridCsrPtrA[offsetPtrS];
		double *csrCofAL=&MultiGridCsrCofA[offsetIndL];
		double *csrCofAS=&MultiGridCsrCofA[offsetIndS];
		int    *csrIndAS=&MultiGridCsrIndA[offsetIndS];
		int    *csrPtrR =&MultiGridCsrPtrR[offsetPtrR];
		int    *csrIndR =&MultiGridCsrIndR[offsetIndR];
		
		#pragma acc kernels present(                   \
			gridCellMinL[0:DIM], gridCellMinS[0:DIM],  \
			gridCountL[0:DIM],gridCountS[0:DIM],       \
			csrPtrAL[0:NL+1],csrPtrAS[0:NS+1],         \
			csrCofAL[0:nnzL],csrCofAS[0:nnzS],         \
			csrIndAS[0:nnzS],                          \
			csrPtrR[0:NL+1], csrIndR[0:NS]             \
		)
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRowL=0;iRowL<NL;++iRowL){
			const int iLGX = (iRowL/(DIM*gridCountL[1]*gridCountL[2])) % gridCountL[0];
			const int iLGY = (iRowL/(DIM*gridCountL[2])) % gridCountL[1];
			const int iLGZ = (iRowL/DIM) % gridCountL[2];
			const int rD   =  iRowL % DIM;
			
			const int cX  = ( 3<gridCountL[0] ? 3:gridCountL[0] );
			const int cY  = ( 3<gridCountL[1] ? 3:gridCountL[1] );
			const int cZ  = ( 3<gridCountL[2] ? 3:gridCountL[2] );
			
			double cofRowL[DIM*POW_3_DIM];
			#pragma acc loop seq
			for(int shiftL=0;shiftL<DIM*POW_3_DIM;++shiftL){
				cofRowL[shiftL]=0.0;
			}
			
			for(int iNonzeroR=csrPtrR[iRowL];iNonzeroR<csrPtrR[iRowL+1];++iNonzeroR){
				const int iRowS = csrIndR[iNonzeroR];
				
				
				#pragma acc loop seq
				for(int iNonzeroS=csrPtrAS[iRowS];iNonzeroS<csrPtrAS[iRowS+1];++iNonzeroS){
					const int iColumnS =csrIndAS[iNonzeroS];
					const int jSGX = (iColumnS/(DIM*gridCountS[1]*gridCountS[2])) % gridCountS[0];
					const int jSGY = (iColumnS/(DIM*gridCountS[2])) % gridCountS[1];
					const int jSGZ = (iColumnS/DIM) % gridCountS[2];
					const int sD   =  iColumnS % DIM;
					
					const int jSCX = jSGX + gridCellMinS[0];
					const int jSCY = jSGY + gridCellMinS[1];
					const int jSCZ = jSGZ + gridCellMinS[2];
					const int jLGX = (jSCX/2) - gridCellMinL[0];
					const int jLGY = (jSCY/2) - gridCellMinL[1];
					const int jLGZ = (jSCZ/2) - gridCellMinL[2];
					const int dLGX = jLGX - iLGX;
					const int dLGY = jLGY - iLGY;
					const int dLGZ = jLGZ - iLGZ;
					const int iX = (dLGX+1+gridCountL[0])%gridCountL[0];
					const int iY = (dLGY+1+gridCountL[1])%gridCountL[1];
					const int iZ = (dLGZ+1+gridCountL[2])%gridCountL[2];
					const int shiftL = DIM*(cZ*cY*iX+cZ*iY+iZ)+sD;
					
					#ifndef _OPENACC
					const int iNonzeroL = csrPtrAL[ iRowL ] + shiftL;
					const int jLG = GridId(jLGX,jLGY,jLGZ,gridCountL);
					const int iColumnL = DIM*jLG+sD;
					assert(MultiGridCsrIndA[ offsetIndL + iNonzeroL ]==iColumnL);
					#endif
					
					cofRowL[ shiftL ] += csrCofAS[ iNonzeroS ];
				}					
			}
			
			#pragma acc loop seq
			for(int shiftL=0;shiftL<(DIM*cZ*cY*cX);++shiftL){
				const int iNonzeroL = csrPtrAL[ iRowL ] + shiftL;
				csrCofAL[ iNonzeroL ] = cofRowL[ shiftL ];
			}
			
		}
	}
}

static void calculateMultiGridInvDiagA( void ){
	
	#pragma acc kernels 
	#pragma acc loop independent
	#pragma omp parallel for 
	for(int iRow=0;iRow<AllGridSizeInvDiagA;++iRow){
		MultiGridInvDiagA[iRow]=-1.0*__LINE__;
	}
	
	for(int iPower=0;iPower<MultiGridDepth;++iPower){
		const int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
		const int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
		const int offsetInvDiagA = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeInvDiagA;
		const int *gridCount = MultiGridCount[iPower];
		const int N    = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		const int nnzA = MultiGridCsrNnzA[iPower];
		int    *csrPtrA = &MultiGridCsrPtrA[offsetPtr];
		double *csrCofA = &MultiGridCsrCofA[offsetInd];
		double *invDiagA= &MultiGridInvDiagA[offsetInvDiagA];
		
		#pragma acc kernels present(gridCount[0:DIM]) present(csrPtrA[0:N+1],csrCofA[0:nnzA],invDiagA[0:N])
		#pragma acc loop independent
		#pragma omp parallel for
		for(int iRow=0;iRow<N;++iRow){
			double sumCof=0.0;
			for(int iNonzero=csrPtrA[iRow];iNonzero<csrPtrA[iRow+1];++iNonzero){
				sumCof+=abs(csrCofA[iNonzero]);
			}
			if(sumCof!=0.0){
				invDiagA[iRow] = 2.0/sumCof;
			}
			else{
				invDiagA[iRow] = 0.0;
			}
		}
	}
}



static void allocateDenssMat(const int N, const int M, double (**&mat)){
	mat = (double **)malloc( N * sizeof(double *) );
	for(int iRow=0;iRow<N;++iRow){
		mat[iRow]=(double *)malloc( M * sizeof(double) );
	}
}

static void setDenssMat(const int N, const int M, const int nnz, const int *csrptr, const int *csrind, const double *csrcof, double **mat){
			fprintf(stderr,"line:%d\n",__LINE__);

	for(int iRow=0;iRow<N;++iRow){
		for(int iColumn=0;iColumn<M;++iColumn){
			mat[iRow][iColumn]=0.0;
		}
	}
		fprintf(stderr,"line:%d\n",__LINE__);

	for(int iRow=0;iRow<N;++iRow){
		for(int iNonzero=csrptr[iRow];iNonzero<csrptr[iRow+1];++iNonzero){
			const int iColumn = csrind[iNonzero];
			mat[iRow][iColumn] += csrcof[iNonzero];
		}
	}
			fprintf(stderr,"line:%d\n",__LINE__);

}

static void writeDenssMat(const int N, const int M, double **mat, const char *filename){
	FILE *fp=fopen(filename,"w");
	for(int iRow=0;iRow<N;++iRow){
		for(int iColumn=0;iColumn<M;++iColumn){
			fprintf(fp,"%e,",mat[iRow][iColumn]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

static void freeDenssMat(const int N, const int M, double **mat){
	for(int iRow=0;iRow<N;++iRow){
		free(mat[iRow]);
	}
	free(mat);
}

static void checkMultiGridMatrix( void ){
	
		
	const int fluidcount = FluidParticleEnd-FluidParticleBegin;
	const int totalGridCount = MultiGridCount[0][0]*MultiGridCount[0][1]*MultiGridCount[0][2];
	
	#pragma acc update host(CsrPtrP2G[0:DIM*totalGridCount+1])
	#pragma acc update host(CsrIndP2G[0:CsrNnzP2G],CsrCofP2G[0:CsrNnzP2G])
	#pragma acc update host(CsrPtrG2P[0:DIM*fluidcount+1])
	#pragma acc update host(CsrIndG2P[0:CsrNnzG2P],CsrCofG2P[0:CsrNnzG2P])
	//#pragma acc update host(MultiGridCsrNnzR[0:MultiGridDepth])
	#pragma acc update host(MultiGridCsrPtrR[0:AllGridSizeCsrPtrR])
	#pragma acc update host(MultiGridCsrIndR[0:AllGridSizeCsrIndR],MultiGridCsrCofR[0:AllGridSizeCsrIndR])
	//#pragma acc update host(MultiGridCsrNnzP[0:MultiGridDepth])
	#pragma acc update host(MultiGridCsrPtrP[0:AllGridSizeCsrPtrP])
	#pragma acc update host(MultiGridCsrIndP[0:AllGridSizeCsrIndP],MultiGridCsrCofP[0:AllGridSizeCsrIndP])
	
	
	
	log_printf("MultiGridDepth=%d\n",MultiGridDepth);
	for(int iPower=0;iPower<MultiGridDepth;++iPower){
		log_printf("MultiGridCellMin[%d]=(%d,%d,%d)\n",iPower, MultiGridCellMin[iPower][0],MultiGridCellMin[iPower][1],MultiGridCellMin[iPower][2]);
		log_printf("MultiGridCellMax[%d]=(%d,%d,%d)\n",iPower, MultiGridCellMax[iPower][0],MultiGridCellMax[iPower][1],MultiGridCellMax[iPower][2]);
		log_printf("MultiGridCount[%d]  =(%d,%d,%d)\n",iPower, MultiGridCount[iPower][0],  MultiGridCount[iPower][1],  MultiGridCount[iPower][2]);
	}
	log_printf("TopGridCount=(%d x %d x %d)=%d\n", TopGridCount[0],TopGridCount[1],TopGridCount[2],TotalTopGridCount);


	fprintf(stderr,"line:%d\n",__LINE__);
	double **denssA;
	// CsrA
	{
		int N = DIM*fluidcount;
		int M = DIM*fluidcount;
		allocateDenssMat(N,M,denssA);
		setDenssMat(N,M,NonzeroCountA,CsrPtrA,CsrIndA,CsrCofA,denssA);
		writeDenssMat(N,M,denssA,"CsrA.csv");
	}
	
	

	// CsrP2G
	double **denssP2G;
	fprintf(stderr,"line:%d\n",__LINE__);
	{
		int N = DIM*MultiGridCount[0][0]*MultiGridCount[0][1]*MultiGridCount[0][2];
		int M = DIM*fluidcount;
		fprintf(stderr,"line:%d\n",__LINE__);
		allocateDenssMat(N,M,denssP2G);
		fprintf(stderr,"line:%d\n",__LINE__);
		setDenssMat(N,M,CsrNnzP2G,CsrPtrP2G,CsrIndP2G,CsrCofP2G,denssP2G);
		fprintf(stderr,"line:%d\n",__LINE__);
		writeDenssMat(N,M,denssP2G,"CsrP2G.csv");
		fprintf(stderr,"line:%d\n",__LINE__);
	}
	
	// CsrG2P
	double **denssG2P;
	fprintf(stderr,"line:%d\n",__LINE__);
	{
		int N = DIM*fluidcount;
		int M = DIM*MultiGridCount[0][0]*MultiGridCount[0][1]*MultiGridCount[0][2];
		allocateDenssMat(N,M,denssG2P);
		setDenssMat(N,M,CsrNnzG2P,CsrPtrG2P,CsrIndG2P,CsrCofG2P,denssG2P);
		writeDenssMat(N,M,denssG2P,"CsrG2P.csv");
	}
	
	double **denssMultiA0;
	// CsrMultiA0
	fprintf(stderr,"line:%d\n",__LINE__);
	{
		const int iPower=0;
		int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
		int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
		const int *gridCount = MultiGridCount[iPower];
		int N = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		int M = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		
		allocateDenssMat(N,M,denssMultiA0);
		setDenssMat(N,M,MultiGridCsrNnzA[iPower],&MultiGridCsrPtrA[offsetPtr],&MultiGridCsrIndA[offsetInd],&MultiGridCsrCofA[offsetInd],denssMultiA0);
		writeDenssMat(N,M,denssMultiA0,"MultiA0.csv");
	}
	
	double **denssMultiR1;
	// CsrMultiR1
	
	fprintf(stderr,"line:%d\n",__LINE__);
	{
		const int iPower=1;
		int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrR + (MultiGridDepth-iPower-1);
		int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndR;
		const int *gridCountL = MultiGridCount[iPower];
		const int *gridCountS = MultiGridCount[iPower-1];
		int N = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
		int M = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
		fprintf(stderr,"line:%d, iPower=%d, MultiGridCsrR[iPower]\n",__LINE__,iPower);
		fprintf(stderr,"MultiGridCsrNnzR[iPower]=%d\n", MultiGridCsrNnzR[iPower]);
		for(int iRow=0;iRow<N;++iRow){
			fprintf(stderr, "%d:(%d,%d)\n", iRow, MultiGridCsrPtrR[offsetPtr+iRow], MultiGridCsrPtrR[offsetPtr+iRow+1]);
		}
		allocateDenssMat(N,M,denssMultiR1);
		setDenssMat(N,M,MultiGridCsrNnzR[iPower],&MultiGridCsrPtrR[offsetPtr],&MultiGridCsrIndR[offsetInd],&MultiGridCsrCofR[offsetInd],denssMultiR1);
		writeDenssMat(N,M,denssMultiR1,"MultiR1.csv");
	}

	double **denssMultiR2;
	// CsrMultiR2
	fprintf(stderr,"line:%d\n",__LINE__);
	if(2<MultiGridDepth){
		const int iPower=2;
		int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrR + (MultiGridDepth-iPower-1);
		int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndR;
		const int *gridCountL = MultiGridCount[iPower];
		const int *gridCountS = MultiGridCount[iPower-1];
		int N = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
		int M = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
		fprintf(stderr,"line:%d, iPower=%d, MultiGridCsrR[iPower]\n",__LINE__,iPower);
		fprintf(stderr,"MultiGridCsrNnzR[iPower]=%d\n", MultiGridCsrNnzR[iPower]);
		for(int iRow=0;iRow<N;++iRow){
			fprintf(stderr, "%d:(%d,%d)\n", iRow, MultiGridCsrPtrR[offsetPtr+iRow], MultiGridCsrPtrR[offsetPtr+iRow+1]);
		}
		allocateDenssMat(N,M,denssMultiR2);
		setDenssMat(N,M,MultiGridCsrNnzR[iPower],&MultiGridCsrPtrR[offsetPtr],&MultiGridCsrIndR[offsetInd],&MultiGridCsrCofR[offsetInd],denssMultiR2);
		writeDenssMat(N,M,denssMultiR2,"MultiR2.csv");
	}

	
	
	
	double **denssMultiP1;
	// CsrMultiP1
	fprintf(stderr,"line:%d\n",__LINE__);
	{
		const int iPower=1;
		int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrP + (MultiGridDepth-iPower-1);
		int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndP;
		const int *gridCountL = MultiGridCount[iPower];
		const int *gridCountS = MultiGridCount[iPower-1];
		int N = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
		int M = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
		allocateDenssMat(N,M,denssMultiP1);
		setDenssMat(N,M,MultiGridCsrNnzP[iPower],&MultiGridCsrPtrP[offsetPtr],&MultiGridCsrIndP[offsetInd],&MultiGridCsrCofP[offsetInd],denssMultiP1);
		writeDenssMat(N,M,denssMultiP1,"MultiP1.csv");
	}
	
	
	double **denssMultiP2;
	// CsrMultiP2
	fprintf(stderr,"line:%d\n",__LINE__);
	if(2<MultiGridDepth){
		const int iPower=2;
		int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrP + (MultiGridDepth-iPower-1);
		int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndP;
		const int *gridCountL = MultiGridCount[iPower];
		const int *gridCountS = MultiGridCount[iPower-1];
		int N = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
		int M = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
		allocateDenssMat(N,M,denssMultiP2);
		setDenssMat(N,M,MultiGridCsrNnzP[iPower],&MultiGridCsrPtrP[offsetPtr],&MultiGridCsrIndP[offsetInd],&MultiGridCsrCofP[offsetInd],denssMultiP2);
		writeDenssMat(N,M,denssMultiP2,"MultiP2.csv");
	}
	
	double **denssMultiA1;
	// CsrMultiA1
	fprintf(stderr,"line:%d\n",__LINE__);
	{
		const int iPower=1;
		int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
		int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
		const int *gridCount = MultiGridCount[iPower];
		int N = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		int M = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		allocateDenssMat(N,M,denssMultiA1);
		setDenssMat(N,M,MultiGridCsrNnzA[iPower],&MultiGridCsrPtrA[offsetPtr],&MultiGridCsrIndA[offsetInd],&MultiGridCsrCofA[offsetInd],denssMultiA1);
		writeDenssMat(N,M,denssMultiA1,"MultiA1.csv");
	}
	
	double **denssMultiA2;
	// CsrMultiA2
	fprintf(stderr,"line:%d\n",__LINE__);
	if(2<MultiGridDepth){
		const int iPower=2;
		int offsetPtr = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
		int offsetInd = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
		const int *gridCount = MultiGridCount[iPower];
		int N = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		int M = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		allocateDenssMat(N,M,denssMultiA2);
		setDenssMat(N,M,MultiGridCsrNnzA[iPower],&MultiGridCsrPtrA[offsetPtr],&MultiGridCsrIndA[offsetInd],&MultiGridCsrCofA[offsetInd],denssMultiA2);
		writeDenssMat(N,M,denssMultiA2,"MultiA2.csv");
	}
	
	fprintf(stderr,"line:%d\n",__LINE__);
	// InvDiagA
	{
		FILE *fp = fopen("InvDiagA.csv","w");
		int N = DIM*fluidcount;
		for(int iRow=0;iRow<N;++iRow){
			fprintf(fp, "%e,\n", InvDiagA[ iRow ]);
		}
		fclose(fp);
	}
	
	
	
	if(0<MultiGridDepth){
		FILE *fp = fopen("InvD0.csv","w");
		const int iPower=0;
		int offset = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeInvDiagA;
		const int *gridCount = MultiGridCount[iPower];
		int N = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		for(int iRow=0;iRow<N;++iRow){
			fprintf(fp, "%e,\n", MultiGridInvDiagA[ offset + iRow ]);
		}
		fclose(fp);
	}
	if(1<MultiGridDepth){
		FILE *fp = fopen("InvD1.csv","w");
		const int iPower=1;
		int offset = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeInvDiagA;
		const int *gridCount = MultiGridCount[iPower];
		int N = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		for(int iRow=0;iRow<N;++iRow){
			fprintf(fp, "%e,\n", MultiGridInvDiagA[ offset + iRow ]);
		}
		fclose(fp);
	}
	if(2<MultiGridDepth){
		FILE *fp = fopen("InvD2.csv","w");
		const int iPower=2;
		int offset = MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeInvDiagA;
		const int *gridCount = MultiGridCount[iPower];
		int N = DIM*gridCount[0]*gridCount[1]*gridCount[2];
		for(int iRow=0;iRow<N;++iRow){
			fprintf(fp, "%e,\n", MultiGridInvDiagA[ offset + iRow ]);
		}
		fclose(fp);
	}	
	//	double **denssR1A0P1;
//	fprintf(stderr,"line:%d\n",__LINE__);
//	{
//		const int iPower=1;
//		const int *gridCountL = MultiGridCount[iPower];
//		const int *gridCountS = MultiGridCount[iPower-1];
//		int NL = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
//		int NS = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
//		allocateDenssMat(NL,NL,denssR1A0P1);
//
//		for(int iP=0;iP<NL;++iP){
//			for(int iQ=0;iQ<NL;++iQ){
//				denssR1A0P1[iP][iQ]=0.0;
//			}
//		}
//
//		for(int iP=0;iP<NL;++iP){
//			for(int iQ=0;iQ<NL;++iQ){
//				for(int iR=0;iR<NS;++iR){
//					for(int iS=0;iS<NS;++iS){
//						double tmp = 1.0;
//						denssR1A0P1[iP][iQ] += denssMultiR1[iP][iR]*denssMultiA0[iR][iS]*denssMultiP1[iS][iQ];
//					}
//				}
//			}
//		}
//		writeDenssMat(NL,NL,denssR1A0P1,"R1A0P1.csv");
//	}
	
//	double **denssP2GAG2P;
//	fprintf(stderr,"line:%d\n",__LINE__);
//	{
//		int NP = DIM*fluidcount;
//		const int iPower=0;
//		const int *gridCount = MultiGridCount[iPower];
//		int NG = DIM*gridCount[0]*gridCount[1]*gridCount[2];
//
//		allocateDenssMat(NG,NG,denssP2GAG2P);
//		for(int iP=0;iP<NG;++iP){
//			for(int iQ=0;iQ<NG;++iQ){
//				denssP2GAG2P[iP][iQ]=0.0;
//			}
//		}
//		
//		for(int iP=0;iP<NG;++iP){
//			for(int iQ=0;iQ<NG;++iQ){
//				for(int iR=0;iR<NP;++iR){
//					for(int iS=0;iS<NP;++iS){
//						denssP2GAG2P[iP][iQ] += denssP2G[iP][iR]*denssA[iR][iS]*denssG2P[iS][iQ];
//					}
//				}
//			}
//		}	
//		writeDenssMat(NG,NG,denssP2GAG2P,"P2GAG2A.csv");
//	}
	
	exit(1);

}

	


static void freeMultiGridMatrix( void ){
	
	free(MultiGridCellMin);
	free(MultiGridCellMax);
	free(MultiGridCount);
	#pragma acc exit data delete(MultiGridCellMin,MultiGridCellMax,MultiGridCount)
	
	free(MultiGridOffset);
	
	free(InvDiagA);
	#pragma acc exit data delete(InvDiagA)
	
	free(CsrPtrP2G);
	free(CsrIndP2G);
	free(CsrCofP2G);
	#pragma acc exit data delete(CsrPtrP2G,CsrIndP2G,CsrCofP2G)

	free(CsrPtrG2P);
	free(CsrIndG2P);
	free(CsrCofG2P);
	#pragma acc exit data delete(CsrPtrG2P,CsrIndG2P,CsrCofG2P)
	
	free(MultiGridVecS);
	free(MultiGridVecR);
	free(MultiGridVecQ);
	#pragma acc exit data delete(MultiGridVecS,MultiGridVecR,MultiGridVecQ)
	
	free(MultiGridCsrNnzA);
	free(MultiGridCsrPtrA);
	free(MultiGridCsrIndA);
	free(MultiGridCsrCofA);
	#pragma acc exit data delete(MultiGridCsrPtrA)
	#pragma acc exit data delete(MultiGridCsrIndA) 
	#pragma acc exit data delete(MultiGridCsrCofA) 
	
	free(MultiGridInvDiagA);
	#pragma acc exit data delete(MultiGridInvDiagA)
	
	free(MultiGridCsrNnzR);
	free(MultiGridCsrPtrR);
	free(MultiGridCsrIndR);
	free(MultiGridCsrCofR);
	#pragma acc exit data delete(MultiGridCsrPtrR,MultiGridCsrIndR,MultiGridCsrCofR)
	
	free(MultiGridCsrNnzP);
	free(MultiGridCsrPtrP);
	free(MultiGridCsrIndP);
	free(MultiGridCsrCofP);
	#pragma acc exit data delete(MultiGridCsrPtrP,MultiGridCsrIndP,MultiGridCsrCofP)
	
}


#ifdef _CUDA
static void mycusparseDcsrmv(cusparseHandle_t cusparse, const int m, const int n, const int nnz, const double alpha, double *csrVal, int *csrRowPtr, int *csrColInd, double *x, const double beta, double *y )
{
	static int init_flag=0;
	static size_t bufferSize=0;
	static void * buffer;
	
	if(init_flag==0){
		bufferSize=8;
		cudaMalloc( &buffer, bufferSize );
		init_flag=1;
	}
	
	cusparseSpMatDescr_t mat;
	cusparseCreateCsr( &mat, m, n, nnz, csrRowPtr, csrColInd, csrVal, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
	cusparseDnVecDescr_t vecX;
	cusparseCreateDnVec( &vecX, n, x, CUDA_R_64F );
	cusparseDnVecDescr_t vecY;
	cusparseCreateDnVec( &vecY, m, y, CUDA_R_64F );
	
	size_t requiredSize;
	cusparseSpMV_bufferSize( cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, &requiredSize);
	if( requiredSize > bufferSize ){
		cudaFree( buffer );
		bufferSize = 2*requiredSize;
		cudaMalloc( &buffer, bufferSize );
	}
	//cudaMalloc( &buffer, requiredSize );
	cusparseSpMV( cusparse, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vecX, &beta, vecY, CUDA_R_64F, CUSPARSE_CSRMV_ALG1, buffer );
	//cudaFree( buffer );
	
	cusparseDestroySpMat(mat);
	cusparseDestroyDnVec(vecX);
	cusparseDestroyDnVec(vecY);
	
}

static void myDcsrmv( cusparseHandle_t cusparse, const int m, const int n, const int nnz, const double alpha, const double *csrVal, const int *csrRowPtr, const int *csrColInd, const double *x, const double beta, double *y)
{
	#pragma acc kernels deviceptr(csrVal,csrRowPtr,csrColInd,x,y)
	//#pragma acc kernels present(csrVal[0:nnz],csrRowPtr[0:m+1],csrColInd[0:nnz],x[0:n],y[0:m])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<m; ++iRow){
		double sum = 0.0;
		#pragma acc loop seq
		for(int iNonZero=csrRowPtr[iRow];iNonZero<csrRowPtr[iRow+1];++iNonZero){
			const int iColumn=csrColInd[iNonZero];
			sum += alpha*csrVal[iNonZero]*x[iColumn];
		}
		y[iRow] *= beta;
		y[iRow] += sum;
	}
}

static void mycublasDdmv( cublasHandle_t cublas, const int n, const double alpha, const double *diag, const double *x, const double beta, double *y )
{
	// cublasDsbmv(cublas, CUBLAS_FILL_MODE_LOWER, n, 0, &alpha, diag, 1, x, 1, &beta, y, 1);
	// This was not good performance when n is very large (> 307,200) (20221107 @ A100(80G) + HPC-SDK 21.5)
	
	#pragma acc kernels deviceptr(diag,x,y)
	//#pragma acc kernels present(diag[0:n],x[0:n],y[0:n])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<n;++iRow){
		y[iRow] *= beta;
		y[iRow] += alpha*diag[iRow]*x[iRow];
	}
}

static void mycublasDscal( cublasHandle_t cublas, const int n, const double alpha, double *x )
{
	cublasDscal( cublas, n, &alpha, x, 1 );
}

static void mycublasDcopy( cublasHandle_t cublas, const int n, const double *x, double *y )
{
	cublasDcopy( cublas, n, x, 1, y, 1);
}

static void mydevDset( const int n, const double alpha, double *x )
{
	#pragma acc kernels deviceptr(x)
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<n;++iRow){
		x[iRow] = alpha;
	}
}


static void preconditionWithMultiGrid(cublasHandle_t cublas, cusparseHandle_t cusparse, double *q, double *s, double *r ){
	cTill = clock(); cImplicitSolve += (cTill-cFrom); cFrom = cTill;
	
	#ifdef TWO_DIMENSIONAL
	const int dim=2;
	#else
	const int dim=3;
	#endif
	
	
	#pragma acc host_data use_device(                         \
		InvDiagA,                                             \
		CsrCofP2G, CsrPtrP2G, CsrIndP2G,                      \
		CsrCofG2P, CsrPtrG2P, CsrIndG2P,                      \
		MultiGridCsrCofR, MultiGridCsrPtrR, MultiGridCsrIndR, \
		MultiGridCsrCofP, MultiGridCsrPtrP, MultiGridCsrIndP, \
		MultiGridCsrCofA, MultiGridCsrPtrA, MultiGridCsrIndA, \
		MultiGridInvDiagA,                                    \
		MultiGridVecQ, MultiGridVecS, MultiGridVecR           \
	)
	{
		
		
		////////////---------- segregated precondition (1st term) M = D^(-1) ----------////////////
		{// D^(-1)
			const int N = DIM*(FluidParticleEnd-FluidParticleBegin);
			cublasDcopy( cublas, N, q, 1, r, 1);
			mycublasDdmv( cublas, N, 1.0, InvDiagA, r, 0.0, s);
		}
		
		////////////---------- second term (multi-grid Jacobi) ----------////////////
		{
			const int iPower=0;
			const int offset         =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeVec;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeInvDiagA;
			const int NP = DIM*(FluidParticleEnd-FluidParticleBegin);
			const int NC = DIM*MultiGridCount[0][0]*MultiGridCount[0][1]*MultiGridCount[0][2];
			
			mycublasDcopy( cublas, NP, q, r);
			myDcsrmv( cusparse, NC, NP, CsrNnzP2G, 1.0, CsrCofP2G, CsrPtrP2G, CsrIndP2G, r, 0.0, &MultiGridVecQ[offset] );
			// mycusparseDcsrmv( cusparse, NC, NP, CsrNnzP2G, 1.0, CsrCofP2G, CsrPtrP2G, CsrIndP2G, r, 0.0, &MultiGridVecQ[offset] );
			
			mycublasDscal( cublas, NC, 0.0, &MultiGridVecS[offset] );
			mycublasDcopy( cublas, NC, &MultiGridVecQ[offset], &MultiGridVecR[offset] );
			for(int iter=0;iter<2;++iter){
				mycublasDdmv( cublas, NC, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offset], 1.0, &MultiGridVecS[offset] );
				mycublasDcopy( cublas, NC, &MultiGridVecQ[offset], &MultiGridVecR[offset] );
				myDcsrmv( cusparse, NC, NC, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offset], 1.0, &MultiGridVecR[offset]);
			}
		}
		
		
		for(int iPower=1;iPower<MultiGridDepth-1;++iPower){
			
			const int offsetS        =MultiGridOffset[MultiGridDepth-(iPower-1)-1]*OneGridSizeVec;
			const int offsetL        =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeVec;
			const int offsetPtrR     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrR + (MultiGridDepth-iPower-1);
			const int offsetIndR     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndR;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeInvDiagA;
			const int *gridCountL    =MultiGridCount[iPower];
			const int *gridCountS    =MultiGridCount[iPower-1];
			const int NS = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
			const int NL = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
			
			myDcsrmv( cusparse, NL, NS, MultiGridCsrNnzR[iPower], 1.0, &MultiGridCsrCofR[offsetIndR], &MultiGridCsrPtrR[offsetPtrR], &MultiGridCsrIndR[offsetIndR], &MultiGridVecR[offsetS], 0.0, &MultiGridVecQ[offsetL] );
			// mycusparseDcsrmv( cusparse, NL, NS, MultiGridCsrNnzR[iPower], 1.0, &MultiGridCsrCofR[offsetIndR], &MultiGridCsrPtrR[offsetPtrR], &MultiGridCsrIndR[offsetIndR], &MultiGridVecR[offsetS], 0.0, &MultiGridVecQ[offsetL] );
			
			mycublasDscal( cublas, NL, 0.0, &MultiGridVecS[offsetL]);
			mycublasDcopy( cublas, NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );
			for(int iter=0;iter<2;++iter){
				mycublasDdmv( cublas, NL, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offsetL], 1.0, &MultiGridVecS[offsetL]);
				mycublasDcopy( cublas, NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );
				myDcsrmv( cusparse, NL, NL, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offsetL], 1.0, &MultiGridVecR[offsetL]);
			}
		}
		
		{
			const int iPower=MultiGridDepth-1;
			const int offsetS        =MultiGridOffset[MultiGridDepth-(iPower-1)-1]*OneGridSizeVec;
			const int offsetL        =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeVec;
			const int offsetPtrR     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrR + (MultiGridDepth-iPower-1);
			const int offsetIndR     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndR;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeInvDiagA;
			const int offsetPtrP     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrP + (MultiGridDepth-iPower-1);
			const int offsetIndP     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndP;
			const int *gridCountL    =MultiGridCount[iPower];
			const int *gridCountS    =MultiGridCount[iPower-1];
			const int NS = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
			const int NL = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
			
			myDcsrmv( cusparse, NL, NS, MultiGridCsrNnzR[iPower], 1.0, &MultiGridCsrCofR[offsetIndR], &MultiGridCsrPtrR[offsetPtrR], &MultiGridCsrIndR[offsetIndR], &MultiGridVecR[offsetS], 0.0, &MultiGridVecQ[offsetL] );
			// mycusparseDcsrmv( cusparse, NL, NS, MultiGridCsrNnzR[iPower], 1.0, &MultiGridCsrCofR[offsetIndR], &MultiGridCsrPtrR[offsetPtrR], &MultiGridCsrIndR[offsetIndR], &MultiGridVecR[offsetS], 0.0, &MultiGridVecQ[offsetL] );
			
			mycublasDscal( cublas, NL, 0.0, &MultiGridVecS[offsetL] );
			mycublasDcopy( cublas, NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );
			for(int iter=0;iter<3;++iter){
				mycublasDdmv( cublas, NL, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offsetL], 1.0, &MultiGridVecS[offsetL]);
				mycublasDcopy( cublas, NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );

				myDcsrmv( cusparse, NL, NL, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offsetL], 1.0, &MultiGridVecR[offsetL]);
			}
			mycublasDdmv( cublas, NL, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offsetL], 1.0, &MultiGridVecS[offsetL]);
			myDcsrmv( cusparse, NS, NL, MultiGridCsrNnzP[iPower], 1.0, &MultiGridCsrCofP[offsetIndP], &MultiGridCsrPtrP[offsetPtrP], &MultiGridCsrIndP[offsetIndP], &MultiGridVecS[offsetL], 1.0, &MultiGridVecS[offsetS] );
			// mycusparseDcsrmv( cusparse, NS, NL, MultiGridCsrNnzP[iPower], 1.0, &MultiGridCsrCofP[offsetIndP], &MultiGridCsrPtrP[offsetPtrP], &MultiGridCsrIndP[offsetIndP], &MultiGridVecS[offsetL], 1.0, &MultiGridVecS[offsetS] );
		}
		
		for(int iPower=MultiGridDepth-2;iPower>0;--iPower){
			const int offsetS        =MultiGridOffset[MultiGridDepth-(iPower-1)-1]*OneGridSizeVec;
			const int offsetL        =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeVec;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeInvDiagA;
			const int offsetPtrP     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrP + (MultiGridDepth-iPower-1);
			const int offsetIndP     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndP;
			const int *gridCountL    =MultiGridCount[iPower];
			const int *gridCountS    =MultiGridCount[iPower-1];
			const int NS = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
			const int NL = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
			
			for(int iter=0;iter<2;++iter){
				mycublasDcopy( cublas, NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );
				myDcsrmv( cusparse, NL, NL, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offsetL], 1.0, &MultiGridVecR[offsetL]);
				mycublasDdmv( cublas, NL, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offsetL], 1.0, &MultiGridVecS[offsetL]);
			}
			myDcsrmv( cusparse, NS, NL, MultiGridCsrNnzP[iPower], 1.0, &MultiGridCsrCofP[offsetIndP], &MultiGridCsrPtrP[offsetPtrP], &MultiGridCsrIndP[offsetIndP], &MultiGridVecS[offsetL], 1.0, &MultiGridVecS[offsetS] );
			// mycusparseDcsrmv( cusparse, NS, NL, MultiGridCsrNnzP[iPower], 1.0, &MultiGridCsrCofP[offsetIndP], &MultiGridCsrPtrP[offsetPtrP], &MultiGridCsrIndP[offsetIndP], &MultiGridVecS[offsetL], 1.0, &MultiGridVecS[offsetS] );
		}
		
		{
			const int iPower=0;
			const int offset         =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeVec;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeInvDiagA;
			const int NP = DIM*(FluidParticleEnd-FluidParticleBegin);
			const int NC = DIM*MultiGridCount[0][0]*MultiGridCount[0][1]*MultiGridCount[0][2];
			
			for(int iter=0;iter<2;++iter){
				mycublasDcopy( cublas, NC, &MultiGridVecQ[offset], &MultiGridVecR[offset] );
				myDcsrmv( cusparse, NC, NC, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offset], 1.0, &MultiGridVecR[offset]);
				mycublasDdmv( cublas, NC, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offset], 1.0, &MultiGridVecS[offset]);
			}
			myDcsrmv( cusparse, NP, NC, CsrNnzG2P, 1.0, CsrCofG2P, CsrPtrG2P, CsrIndG2P, &MultiGridVecS[offset], 1.0, s );
			// mycusparseDcsrmv( cusparse, NP, NC, CsrNnzG2P, 1.0, CsrCofG2P, CsrPtrG2P, CsrIndG2P, &MultiGridVecS[offset], 1.0, s );
		}
	}
	
	cTill = clock(); cPrecondition += (cTill-cFrom); cFrom = cTill;
}

static void solveWithConjugatedGradient(void){
	
	const int fluidcount=FluidParticleEnd-FluidParticleBegin;
	const int N = DIM*fluidcount;
	
	double *b = VectorB;
	double *x = (double *)malloc( N*sizeof(double) );
	double *r = (double *)malloc( N*sizeof(double) );
	double *s = (double *)malloc( N*sizeof(double) );
	double *y = (double *)malloc( N*sizeof(double) );
	double *p = (double *)malloc( N*sizeof(double) );
	double *q = (double *)malloc( N*sizeof(double) );
	double *u = (double *)malloc( N*sizeof(double) );
	double *buf = (double *)malloc( N*sizeof(double) );
	double rho=0.0;
	double rhop=0.0;
	double tmp=0.0;
	double alpha=0.0;
	double beta=0.0;
	double rr=0.0;
	double rr0=0.0;
	double rs=0.0;
	double rs0=0.0;
	int iter=0;
	
	const double one=1.0;
	const double minus_one=-1.0;
	const double zero=0.0;	
	
	#pragma acc enter data create(x[0:N],r[0:N],s[0:N],y[0:N],p[0:N],q[0:N],u[0:N],buf[0:N])
	
	// cuda init
	cublasHandle_t cublas;
	cublasCreate(&cublas);
	
	cusparseHandle_t cusparse;
	cusparseCreate(&cusparse);
	
	
	// set intial solution
	#pragma acc kernels present(Velocity[0:ParticleCount][0:DIM],Force[0:ParticleCount][0:DIM],Mass[0:ParticleCount],x[0:N])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iP=0;iP<fluidcount;++iP){
		#pragma acc loop seq
		for(int rD=0;rD<DIM;++rD){
			const int iRow = DIM*iP+rD;
			x[iRow]=Velocity[iP][rD]-Force[iP][rD]/Mass[iP]*Dt;
		}
	}
	
	#pragma acc host_data use_device(b,CsrCofA,CsrPtrA,CsrIndA,x,r,s,y,p,q,u,buf)
	{
		// intialize
		mydevDset( N, -1.0*__LINE__, r );
		mydevDset( N, -1.0*__LINE__, s );
		mydevDset( N, -1.0*__LINE__, y );
		mydevDset( N, -1.0*__LINE__, p );
		mydevDset( N, -1.0*__LINE__, q );
		mydevDset( N, -1.0*__LINE__, u );
		mydevDset( N, -1.0*__LINE__, buf );
		
		#pragma acc host_data use_device(MultiGridVecS,MultiGridVecR,MultiGridVecQ)
		{
			mydevDset( AllGridSizeVecS, -1.0*__LINE__, MultiGridVecS );
			mydevDset( AllGridSizeVecR, -1.0*__LINE__, MultiGridVecR );
			mydevDset( AllGridSizeVecQ, -1.0*__LINE__, MultiGridVecQ );
		}
		
		preconditionWithMultiGrid(cublas,cusparse, b, s, buf );
		//cublasDcopy(cublas,N,b,1,s,1);
		cublasDdot(cublas,N,b,1,s,1,&rs0);
		cublasDdot(cublas,N,b,1,b,1,&rr0);
		
		cublasDcopy(cublas,N,b,1,r,1);
		mycusparseDcsrmv(cusparse,N,N,NonzeroCountA,-1.0,CsrCofA,CsrPtrA,CsrIndA,x,1.0,r);
		preconditionWithMultiGrid(cublas,cusparse, r, s, buf );
		//cublasDcopy(cublas,N,r,1,s,1);
		
		for(iter=0;iter<N;++iter){
			cublasDdot(cublas,N,r,1,r,1,&rr);
			if(rr/rr0 <1.0e-12)break;
			{
				#ifdef CONVERGENCE_CHECK
				cublasDdot(cublas,N,r,1,s,1,&rs);
				log_printf("line:%d, iter,=%d, rr0=,%e, rr=,%e, rs0=,%e, rs=,%e\n",__LINE__,iter,rr0,rr,rs0,rs);
				#endif
			}
			mycusparseDcsrmv(cusparse,N,N,NonzeroCountA,1.0,CsrCofA,CsrPtrA,CsrIndA,s,0.0,y);
			cublasDdot(cublas,N,s,1,y,1,&rho);
			if(iter==0){
				cublasDcopy(cublas,N,s,1,p,1);
				mycusparseDcsrmv(cusparse,N,N,NonzeroCountA,1.0,CsrCofA,CsrPtrA,CsrIndA,p,0.0,q);
			}
			else{
				beta=rho/rhop;
				cublasDscal(cublas,N,&beta,p,1);
				cublasDaxpy(cublas,N,&one,s,1,p,1);
				cublasDscal(cublas,N,&beta,q,1);
				cublasDaxpy(cublas,N,&one,y,1,q,1);
			}
			preconditionWithMultiGrid(cublas,cusparse, q, u, buf ); 
			//cublasDcopy(cublas,N,q,1,u,1);
			cublasDdot(cublas,N,q,1,u,1,&tmp);
			alpha =rho/tmp;
			cublasDaxpy(cublas,N,&alpha,p,1,x,1);
			alpha*=-1;
			cublasDaxpy(cublas,N,&alpha,q,1,r,1);
			cublasDaxpy(cublas,N,&alpha,u,1,s,1);
			rhop=rho;
		}
		cublasDdot(cublas,N,r,1,r,1,&rr);
		cublasDdot(cublas,N,r,1,s,1,&rs);
		log_printf("line:%d, iter,=%d, rr0=,%e, rr=,%e, rs0=,%e, rs=,%e\n",__LINE__,iter,rr0,rr,rs0,rs);
		
		{
			#ifdef CONVERGENCE_CHECK
			cublasDcopy(cublas,N,b,1,r,1);
			mycusparseDcsrmv(cusparse,N,N,NonzeroCountA,-1.0,CsrCofA,CsrPtrA,CsrIndA,x,1.0,r);
			preconditionWithMultiGrid(cublas,cusparse, r, s, buf );
			//cublasDcopy(cublas,N,r,1,s,1);
			cublasDdot(cublas,N,r,1,r,1,&rr);
			cublasDdot(cublas,N,r,1,s,1,&rs);
			log_printf("line:%d, iter,=%d, rr0=,%e, rr=,%e, rs0=,%e, rs=,%e\n",__LINE__,iter,rr0,rr,rs0,rs);
			#endif
		}

	}

	
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
	free(s);
	free(y);
	free(p);
	free(q);
	free(u);
	free(buf);
	#pragma acc exit data delete(x[0:N],r[0:N],s[0:N],y[0:N],p[0:N],q[0:N],u[0:N],buf[0:N])
}


#else //_CUDA is not defined,
static void myDcsrmv( const int m, const int n, const int nnz, const double alpha, const double *csrVal, const int *csrRowPtr, const int *csrColInd, const double *x, const double beta, double *y)
{
	#pragma acc kernels deviceptr(csrVal,csrRowPtr,csrColInd,x,y)
	//#pragma acc kernels present(csrVal[0:nnz],csrRowPtr[0:m+1],csrColInd[0:nnz],x[0:n],y[0:m])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<m; ++iRow){
		double sum = 0.0;
		#pragma acc loop seq
		for(int iNonZero=csrRowPtr[iRow];iNonZero<csrRowPtr[iRow+1];++iNonZero){
			const int iColumn=csrColInd[iNonZero];
			sum += alpha*csrVal[iNonZero]*x[iColumn];
		}
		y[iRow] *= beta;
		y[iRow] += sum;
	}
}

static void myDcsrmvForA( const int m, const int n, const int nnz, const double alpha, const double *csrVal, const int *csrRowPtr, const int *csrColInd, const double *x, const double beta, double *y)
{
	#pragma acc kernels deviceptr(csrVal,csrRowPtr,csrColInd,x,y)
	//#pragma acc kernels present(csrVal[0:nnz],csrRowPtr[0:m+1],csrColInd[0:nnz],x[0:n],y[0:m])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<m; ++iRow){
		double sum = 0.0;
		//#pragma acc loop seq
		#pragma acc loop reduction(+:sum) vector
		for(int iNonZero=csrRowPtr[iRow];iNonZero<csrRowPtr[iRow+1];++iNonZero){
			const int iColumn=csrColInd[iNonZero];
			sum += alpha*csrVal[iNonZero]*x[iColumn];
		}
		y[iRow] *= beta;
		y[iRow] += sum;
	}
}

static void myDdmv( const int n, const double alpha, const double *diag, const double *x, const double beta, double *y )
{
	#pragma acc kernels deviceptr(diag,x,y)
	//#pragma acc kernels present(diag[0:n],x[0:n],y[0:n])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<n;++iRow){
		y[iRow] *= beta;
		y[iRow] += alpha*diag[iRow]*x[iRow];
	}
}

static void myDdot( const int n, const double *x, const double *y, double *res )
{
	double sum=0.0;
	#pragma acc kernels copy(sum) deviceptr(x,y)
	//#pragma acc kernels copy(sum) present(x[0:n],y[0:n])
	#pragma acc loop reduction(+:sum)
	#pragma omp parallel for reduction(+:sum)
	for(int iRow=0;iRow<n;++iRow){
		sum += x[iRow]*y[iRow];
	}	
	(*res)=sum;
}

static void myDaxpy( const int n, const double alpha, const double *x, double *y )
{
	#pragma acc kernels deviceptr(x,y)
	//#pragma acc kernels present(x[0:n],y[0:n])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<n;++iRow){
		y[iRow] += alpha*x[iRow];
	}
}

static void myDscal( const int n, const double alpha, double *x )
{
	#pragma acc kernels deviceptr(x)
	//#pragma acc kernels present(x[0:n])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<n;++iRow){
		x[iRow] *= alpha;
	}
}

static void myDcopy( const int n, const double *x, double *y )
{
	#pragma acc kernels deviceptr(x,y)
	//#pragma acc kernels present(x[0:n],y[0:n])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<n;++iRow){
		y[iRow] = x[iRow];
	}
}

static void myDset( const int n, const double alpha, double *x )
{
	#pragma acc kernels deviceptr(x)
	//#pragma acc kernels present(x[0:n])
	#pragma acc loop independent
	#pragma omp parallel for
	for(int iRow=0;iRow<n;++iRow){
		x[iRow] = alpha;
	}
}


static void preconditionWithMultiGrid( const double *q, double *s, double *r ){ // "r" is N memory allocated buffer
	cTill = clock(); cImplicitSolve += (cTill-cFrom); cFrom = cTill;
	
	#ifdef TWO_DIMENSIONAL
	const int dim=2;
	#else
	const int dim=3;
	#endif
	
	#pragma acc host_data use_device(                         \
		InvDiagA,                                             \
		CsrCofP2G, CsrPtrP2G, CsrIndP2G,                      \
		CsrCofG2P, CsrPtrG2P, CsrIndG2P,                      \
		MultiGridCsrCofR, MultiGridCsrPtrR, MultiGridCsrIndR, \
		MultiGridCsrCofP, MultiGridCsrPtrP, MultiGridCsrIndP, \
		MultiGridCsrCofA, MultiGridCsrPtrA, MultiGridCsrIndA, \
		MultiGridInvDiagA,                                    \
		MultiGridVecQ, MultiGridVecS, MultiGridVecR           \
	)
	{
		
		////////////---------- segregated precondition (1st term) M = D^(-1) ----------////////////
		{// D^(-1)
			const int N = DIM*(FluidParticleEnd-FluidParticleBegin);		
			myDcopy( N, q, r );
			myDdmv( N, 1.0, InvDiagA, r, 0.0, s);
		}
		
		
		////////////---------- second term (multi-grid Jacobi) ----------////////////
		{
			const int iPower=0;
			const int offset         =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeVec;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeInvDiagA;
			const int NP = DIM*(FluidParticleEnd-FluidParticleBegin);
			const int NC = DIM*MultiGridCount[0][0]*MultiGridCount[0][1]*MultiGridCount[0][2];
			
			myDcopy( NP, q, r );
			myDcsrmv( NC, NP, CsrNnzP2G, 1.0, CsrCofP2G, CsrPtrP2G, CsrIndP2G, r, 0.0, &MultiGridVecQ[offset] );
			myDscal( NC, 0.0, &MultiGridVecS[offset] );
			myDcopy( NC, &MultiGridVecQ[offset], &MultiGridVecR[offset] );
			for(int iter=0;iter<2;++iter){
				myDdmv( NC, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offset], 1.0, &MultiGridVecS[offset]);
				myDcopy( NC, &MultiGridVecQ[offset], &MultiGridVecR[offset] );
				myDcsrmv( NC, NC, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offset], 1.0, &MultiGridVecR[offset]);
			}
		}
		
		for(int iPower=1;iPower<MultiGridDepth-1;++iPower){
			const int offsetS        =MultiGridOffset[MultiGridDepth-(iPower-1)-1]*OneGridSizeVec;
			const int offsetL        =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeVec;
			const int offsetPtrR     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrR + (MultiGridDepth-iPower-1);
			const int offsetIndR     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndR;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeInvDiagA;
			const int *gridCountL    =MultiGridCount[iPower];
			const int *gridCountS    =MultiGridCount[iPower-1];
			const int NS = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
			const int NL = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
			
			myDcsrmv( NL, NS, MultiGridCsrNnzR[iPower], 1.0, &MultiGridCsrCofR[offsetIndR], &MultiGridCsrPtrR[offsetPtrR], &MultiGridCsrIndR[offsetIndR], &MultiGridVecR[offsetS], 0.0, &MultiGridVecQ[offsetL] );
			
			myDscal( NL, 0.0, &MultiGridVecS[offsetL] );
			myDcopy( NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );
			for(int iter=0;iter<2;++iter){
				myDdmv( NL, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offsetL], 1.0, &MultiGridVecS[offsetL]);
				myDcopy( NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );
				myDcsrmv( NL, NL, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offsetL], 1.0, &MultiGridVecR[offsetL]);
			}
		}
		
		{
			const int iPower=MultiGridDepth-1;
			const int offsetS        =MultiGridOffset[MultiGridDepth-(iPower-1)-1]*OneGridSizeVec;
			const int offsetL        =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeVec;
			const int offsetPtrR     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrR + (MultiGridDepth-iPower-1);
			const int offsetIndR     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndR;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeInvDiagA;
			const int offsetPtrP     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrP + (MultiGridDepth-iPower-1);
			const int offsetIndP     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndP;
			const int *gridCountL    =MultiGridCount[iPower];
			const int *gridCountS    =MultiGridCount[iPower-1];
			const int NS = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
			const int NL = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
			
			myDcsrmv( NL, NS, MultiGridCsrNnzR[iPower], 1.0, &MultiGridCsrCofR[offsetIndR], &MultiGridCsrPtrR[offsetPtrR], &MultiGridCsrIndR[offsetIndR], &MultiGridVecR[offsetS], 0.0, &MultiGridVecQ[offsetL] );
			
			myDscal( NL, 0.0, &MultiGridVecS[offsetL] );
			myDcopy( NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );
			for(int iter=0;iter<3;++iter){
				myDdmv( NL, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offsetL], 1.0, &MultiGridVecS[offsetL]);
				myDcopy( NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );
				myDcsrmv( NL, NL, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offsetL], 1.0, &MultiGridVecR[offsetL]);
			}
			myDdmv( NL, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offsetL], 1.0, &MultiGridVecS[offsetL]);
			myDcsrmv( NS, NL, MultiGridCsrNnzP[iPower], 1.0, &MultiGridCsrCofP[offsetIndP], &MultiGridCsrPtrP[offsetPtrP], &MultiGridCsrIndP[offsetIndP], &MultiGridVecS[offsetL], 1.0, &MultiGridVecS[offsetS] );
		}
		
		for(int iPower=MultiGridDepth-2;iPower>0;--iPower){
			const int offsetS        =MultiGridOffset[MultiGridDepth-(iPower-1)-1]*OneGridSizeVec;
			const int offsetL        =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeVec;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeInvDiagA;
			const int offsetPtrP     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrPtrP + (MultiGridDepth-iPower-1);
			const int offsetIndP     =MultiGridOffset[MultiGridDepth-(iPower  )-1]*OneGridSizeCsrIndP;
			const int *gridCountL    =MultiGridCount[iPower];
			const int *gridCountS    =MultiGridCount[iPower-1];
			const int NS = DIM*gridCountS[0]*gridCountS[1]*gridCountS[2];
			const int NL = DIM*gridCountL[0]*gridCountL[1]*gridCountL[2];
			
			for(int iter=0;iter<2;++iter){
				myDcopy( NL, &MultiGridVecQ[offsetL], &MultiGridVecR[offsetL] );
				myDcsrmv( NL, NL, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offsetL], 1.0, &MultiGridVecR[offsetL]);
				myDdmv( NL, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offsetL], 1.0, &MultiGridVecS[offsetL]);
			}
			myDcsrmv( NS, NL, MultiGridCsrNnzP[iPower], 1.0, &MultiGridCsrCofP[offsetIndP], &MultiGridCsrPtrP[offsetPtrP], &MultiGridCsrIndP[offsetIndP], &MultiGridVecS[offsetL], 1.0, &MultiGridVecS[offsetS] );
		}
		
		{
			const int iPower=0;
			const int offset         =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeVec;
			const int offsetPtrA     =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrPtrA + (MultiGridDepth-iPower-1);
			const int offsetIndA     =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeCsrIndA;
			const int offsetInvDiagA =MultiGridOffset[MultiGridDepth-iPower-1]*OneGridSizeInvDiagA;
			const int NP = DIM*(FluidParticleEnd-FluidParticleBegin);
			const int NC = DIM*MultiGridCount[0][0]*MultiGridCount[0][1]*MultiGridCount[0][2];
			
			for(int iter=0;iter<2;++iter){
				myDcopy( NC, &MultiGridVecQ[offset], &MultiGridVecR[offset] );
				myDcsrmv( NC, NC, MultiGridCsrNnzA[iPower], -1.0, &MultiGridCsrCofA[offsetIndA], &MultiGridCsrPtrA[offsetPtrA], &MultiGridCsrIndA[offsetIndA], &MultiGridVecS[offset], 1.0, &MultiGridVecR[offset]);
				myDdmv( NC, 1.0, &MultiGridInvDiagA[offsetInvDiagA], &MultiGridVecR[offset], 1.0, &MultiGridVecS[offset]);
			}
			myDcsrmv( NP, NC, CsrNnzG2P, 1.0, CsrCofG2P, CsrPtrG2P, CsrIndG2P, &MultiGridVecS[offset], 1.0, s );
			
		}
	}
	
	cTill = clock(); cPrecondition += (cTill-cFrom); cFrom = cTill;
}



static void solveWithConjugatedGradient(void){
	const int fluidcount = FluidParticleEnd-FluidParticleBegin;
	const int N = DIM*fluidcount;
	
	const double *b = VectorB;
	double *x = (double *)malloc( N*sizeof(double) );
	double *r = (double *)malloc( N*sizeof(double) );
	double *s = (double *)malloc( N*sizeof(double) );
	double *y = (double *)malloc( N*sizeof(double) );
	double *p = (double *)malloc( N*sizeof(double) );
	double *q = (double *)malloc( N*sizeof(double) );
	double *u = (double *)malloc( N*sizeof(double) );
	double *buf = (double *)malloc( N*sizeof(double) );
	double rho=1.0;
	double rhop=0.0;
	double tmp=0.0;
	double alpha=0.0;
	double beta=0.0;
	double rr=0.0;
	double rr0=0.0;
	double rs=0.0;
	double rs0=0.0;
	int iter=0;
	
	
	#pragma acc enter data create(x[0:N],r[0:N],s[0:N],y[0:N],p[0:N],q[0:N],u[0:N],buf[0:N])
	
	
	// set initial solution
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
	
	
	#pragma acc host_data use_device(b,CsrCofA,CsrPtrA,CsrIndA,x,r,s,y,p,q,u,buf)
	{
		// intialize
		myDset( N, -1.0*__LINE__, r );
		myDset( N, -1.0*__LINE__, s );
		myDset( N, -1.0*__LINE__, y );
		myDset( N, -1.0*__LINE__, p );
		myDset( N, -1.0*__LINE__, q );
		myDset( N, -1.0*__LINE__, u );
		myDset( N, -1.0*__LINE__, buf );
		
		#pragma acc host_data use_device(MultiGridVecS,MultiGridVecR,MultiGridVecQ)
		{
			myDset( AllGridSizeVecS, -1.0*__LINE__, MultiGridVecS );
			myDset( AllGridSizeVecR, -1.0*__LINE__, MultiGridVecR );
			myDset( AllGridSizeVecQ, -1.0*__LINE__, MultiGridVecQ );
		}
		
		preconditionWithMultiGrid( b, s, buf );  
		//	myDcopy( N, b, s ); //
		myDdot( N, b, s, &rs0 );
		myDdot( N, b, b, &rr0 );
		if(!(rs0==rs0)){
			log_printf("line:%d, rs0=%e\n",__LINE__,rs0);
			checkMultiGridMatrix();
		}
		
		myDcopy( N, b, r );	
		myDcsrmvForA( N, N, NonzeroCountA, -1.0, CsrCofA, CsrPtrA, CsrIndA, x, 1.0, r );
		preconditionWithMultiGrid( r, s, buf );  //
		//	myDcopy( N, r, s ); //
		
		for(iter=0;iter<N;++iter){
			myDdot( N, r, r, &rr );
			if(rr/rr0 <1.0e-12)break;
			
			#ifdef CONVERGENCE_CHECK
			myDdot( N, r, s, &rs );
			log_printf("line:%d, iter,=%d, rr0=,%e, rr=,%e, rs0=,%e, rs=,%e\n",__LINE__,iter,rr0,rr,rs0,rs);
			#endif
			
			myDcsrmvForA( N, N, NonzeroCountA, 1.0, CsrCofA, CsrPtrA, CsrIndA, s, 0.0, y);
			myDdot( N, s, y, &rho);
			if(iter==0){
				myDcopy( N, s, p );
				myDcsrmvForA( N, N, NonzeroCountA, 1.0, CsrCofA, CsrPtrA, CsrIndA, p, 0.0, q);
			}
			else{
				beta=rho/rhop;
				myDscal( N, beta, p );
				myDaxpy( N, 1.0, s, p );
				myDscal( N, beta, q );
				myDaxpy( N, 1.0, y, q );
			}
			preconditionWithMultiGrid( q, u, buf ); //
			//	myDcopy( N, q, u ); //
			myDdot( N, q, u, &tmp );
			alpha =rho/tmp;
			myDaxpy( N, alpha, p, x );
			myDaxpy( N,-alpha, q, r );
			myDaxpy( N,-alpha, u, s );
			rhop=rho;
		}
		myDdot( N, r, r, &rr );
		myDdot( N, r, s, &rs );
	}
	log_printf("line:%d, iter,=%d, rr0=,%e, rr=,%e, rs0=,%e, rs=,%e\n",__LINE__,iter,rr0,rr,rs0,rs);
	
	#ifdef CONVERGENCE_CHECK
	#pragma acc host_data use_device(b,CsrCofA,CsrPtrA,CsrIndA,x,r,s,y,p,q,u,buf)
	{
		myDcopy( N, b, r );	
		myDcsrmvForA( N, N, NonzeroCountA, -1.0, CsrCofA, CsrPtrA, CsrIndA, x, 1.0, r );
		preconditionWithMultiGrid( r, s, buf );  //
		//	myDcopy( N, b, s ); //
		myDdot( N, r, r, &rr );
		myDdot( N, r, s, &rs );
	}
	log_printf("line:%d, iter,=%d, rr0=,%e, rr=,%e, rs0=,%e, rs=,%e\n",__LINE__,iter,rr0,rr,rs0,rs);
	#endif
	
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
	free(s);
	free(y);
	free(p);
	free(q);
	free(u);
	free(buf);
	#pragma acc exit data delete(x[0:N],r[0:N],s[0:N],y[0:N],p[0:N],q[0:N],u[0:N],buf[0:N])

}

#endif




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

