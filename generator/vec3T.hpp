#ifndef __VEC3T_HPP__INCLUDED__
#define __VEC3T_HPP__INCLUDED__

#include <cmath>



template <typename T>
class vec3;
template <typename T>
class mat3;

template <typename T> int operator== (const vec3<T>& vl, const vec3<T>& vr);
template <typename T> int operator!= (const vec3<T>& vl, const vec3<T>& vr);
template <typename T> int operator < (const vec3<T>& vl, const vec3<T>& vr);
template <typename T> int operator<= (const vec3<T>& vl, const vec3<T>& vr);
template <typename T> int operator== (const mat3<T>& ml, const mat3<T>& mr);
template <typename T> int operator!= (const mat3<T>& ml, const mat3<T>& mr);
template <typename T> vec3<T> operator * (const mat3<T>& ml, const vec3<T>& vr);
template <typename T> vec3<T> operator * (const vec3<T>& vl, const mat3<T>& mr);
template <typename T> T operator * (const vec3<T>& vl, const vec3<T>& vr);
template <typename T> mat3<T> operator * (const mat3<T>& ml, const mat3<T>& mr);
template <typename T> mat3<T> operator % (const vec3<T>& vl, const vec3<T>& vr);
template <typename T> vec3<T> cross(const vec3<T>& vl, const vec3<T>& vr);


template <typename T>
class vec3{
protected:
 union{
  struct{T b0, b1, b2;};
  T b[3];
 };

public:
 vec3(){}
 vec3(const vec3& v):b0(v.b[0]), b1(v.b[1]), b2(v.b[2]){}
 template<typename U> explicit vec3(const vec3<U>& v):b0((T)v.b[0]), b1((T)v.b[1]), b2((T)v.b[2]){}
 vec3(const T& v0, const T& v1, const T& v2):b0(v0), b1(v1), b2(v2){}
    template<typename U> explicit vec3(const U * const v):b0((T)v[0]), b1((T)v[1]), b2((T)v[2]){}
 ~vec3(){}

 template<typename U> vec3& operator = ( const vec3<U>& v ){ b[0] =v.b[0]; b[1] =v.b[1]; b[2] =v.b[2]; return *this; }
 template<typename U> vec3& operator += ( const vec3<U>& v ){ b[0]+=v.b[0]; b[1]+=v.b[1]; b[2]+=v.b[2]; return *this; }
 template<typename U> vec3& operator -= ( const vec3<U>& v ){ b[0]-=v.b[0]; b[1]-=v.b[1]; b[2]-=v.b[2]; return *this; }
 vec3& operator *= ( const T & f ){ b[0]*=f; b[1]*=f; b[2]*=f; return *this; }
 vec3& operator /= ( const T & f ){ b[0]/=f; b[1]/=f; b[2]/=f; return *this; }
 template<typename U> vec3& operator = ( const U * const v){ b[0]=v[0]; b[1]=v[1]; b[2]=v[2]; return *this; }
 template<typename U> vec3& operator += ( const U * const v){ b[0]+=v[0]; b[1]+=v[1]; b[2]+=v[2]; return *this; }
 template<typename U> vec3& operator -= ( const U * const v){ b[0]-=v[0]; b[1]-=v[1]; b[2]-=v[2]; return *this; }
 vec3& set(const T& v0, const T& v1, const T& v2){ b[0]=v0; b[1]=v1; b[2]=v2; return *this; }
 vec3& plus(const T& v0, const T& v1, const T& v2){ b[0]+=v0; b[1]+=v1; b[2]+=v2; return *this; }
 vec3& minus(const T& v0, const T& v1, const T& v2){ b[0]-=v0; b[1]-=v1; b[2]-=v2; return *this; }
 const T& operator [](int i)const{ return b[i]; }
 T& operator[](int i){ return b[i]; }
 const T * c_array(){ return &b[0]; }
 T length(void)const{ return std::sqrt(length2()); }
 T length2(void)const{ return (*this)*(*this); }

 friend class vec3<double>;
 friend class vec3<float>;
 friend class vec3<long>;
 friend class vec3<int>;

 friend int operator==<T> (const vec3 & vl, const vec3 & vr);
 friend int operator!=<T> (const vec3 & vl, const vec3 & vr);
 friend int operator< <T> (const vec3 & vl, const vec3 & vr);
 friend int operator<=<T> (const vec3 & vl, const vec3 & vr);
 friend vec3 operator *<T> (const mat3<T>& ml, const vec3 & vr);
 friend vec3 operator *<T> (const vec3 & vl, const mat3<T>& mr);
 friend T operator *<T> (const vec3 & vl, const vec3 & vr);
 friend mat3<T> operator %<T> (const vec3 & vl, const vec3 & vr);
 friend vec3<T> cross<T> (const vec3 & vl, const vec3 & vr);

};


template <typename T>
class mat3{
protected:
 union{
  struct { T a00, a01, a02, a10, a11, a12, a20, a21, a22; };
  T a[3][3];
 };

public:
 mat3(){}
 mat3(const mat3& m):a00(m.a[0][0]), a01(m.a[0][1]), a02(m.a[0][2]), a10(m.a[1][0]), a11(m.a[1][1]), a12(m.a[1][2]), a20(m.a[2][0]), a21(m.a[2][1]), a22(m.a[2][2]){}
 template<typename U> explicit mat3(const mat3<U>& m):a00((T)m.a[0][0]), a01((T)m.a[0][1]), a02((T)m.a[0][2]), a10((T)m.a[1][0]), a11((T)m.a[1][1]), a12((T)m.a[1][2]), a20((T)m.a[2][0]), a21((T)m.a[2][1]), a22((T)m.a[2][2]){}
 mat3(const T& m00, const T& m01, const T& m02, const T& m10, const T& m11, const T& m12, const T& m20, const T& m21, const T& m22):a00(m00), a01(m01), a02(m02), a10(m10), a11(m11), a12(m12), a20(m20), a21(m21), a22(m22){}
 template<typename U> explicit mat3(const U * const m):a00((T)m[3*0 +0]), a01((T)m[3*0 +1]), a02((T)m[3*0 +2]), a10((T)m[3*1 +0]), a11((T)m[3*1 +1]), a12((T)m[3*1 +2]), a20((T)m[3*2 +0]), a21((T)m[3*2 +1]), a22((T)m[3*2 +2]){}
 mat3(const T& s):a00((T)s), a11((T)s), a22((T)s){}

 template<typename U> mat3& operator = ( const mat3<U>& m ){ a[0][0]=m.a[0][0]; a[0][1]=m.a[0][1]; a[0][2]=m.a[0][2]; a[1][0]=m.a[1][0]; a[1][1]=m.a[1][1]; a[1][2]=m.a[1][2]; a[2][0]=m.a[2][0]; a[2][1]=m.a[2][1]; a[2][2]=m.a[2][2]; return *this; }
 template<typename U> mat3& operator += ( const mat3<U>& m ){ a[0][0]+=m.a[0][0]; a[0][1]+=m.a[0][1]; a[0][2]+=m.a[0][2]; a[1][0]+=m.a[1][0]; a[1][1]+=m.a[1][1]; a[1][2]+=m.a[1][2]; a[2][0]+=m.a[2][0]; a[2][1]+=m.a[2][1]; a[2][2]+=m.a[2][2]; return *this; }
 template<typename U> mat3& operator -= ( const mat3<U>& m ){ a[0][0]-=m.a[0][0]; a[0][1]-=m.a[0][1]; a[0][2]-=m.a[0][2]; a[1][0]-=m.a[1][0]; a[1][1]-=m.a[1][1]; a[1][2]-=m.a[1][2]; a[2][0]-=m.a[2][0]; a[2][1]-=m.a[2][1]; a[2][2]-=m.a[2][2]; return *this; }
 template<typename U> mat3& operator *= ( const mat3<U>& m ){ return *this = *this * (mat3<T>)m;}
 mat3& operator = ( const T& s){
  a[0][0]=0; a[0][1]=0; a[0][2]=0; a[1][0]=0; a[1][1]=0; a[1][2]=0; a[2][0]=0; a[2][1]=0; a[2][2]=0;
  a[0][0]=s; a[1][1]=s; a[2][2]=s;
  return *this;
 }
 mat3& operator += ( const T& s){ a[0][0]+=s; a[1][1]+=s; a[2][2]+=s; return *this; }
 mat3& operator -= ( const T& s){ a[0][0]-=s; a[1][1]-=s; a[2][2]-=s; return *this; }
 mat3& operator *= ( const T& s){ a[0][0]*=s; a[0][1]*=s; a[0][2]*=s; a[1][0]*=s; a[1][1]*=s; a[1][2]*=s; a[2][0]*=s; a[2][1]*=s; a[2][2]*=s; return *this; }
 mat3& operator /= ( const T& s){ a[0][0]/=s; a[0][1]/=s; a[0][2]/=s; a[1][0]/=s; a[1][1]/=s; a[1][2]/=s; a[2][0]/=s; a[2][1]/=s; a[2][2]/=s; return *this; }
 template<typename U> mat3& operator = ( const U * const f ){ a00=f[3*0 +0]; a01=f[3*0 +1]; a02=f[3*0 +2]; a10=f[3*1 +0]; a11=f[3*1 +1]; a12=f[3*1 +2]; a20=f[3*2 +0]; a21=f[3*2 +1]; a22=f[3*2 +2]; return *this; }
 template<typename U> mat3& operator += ( const U * const f ){ a00+=f[3*0 +0]; a01+=f[3*0 +1]; a02+=f[3*0 +2]; a10+=f[3*1 +0]; a11+=f[3*1 +1]; a12+=f[3*1 +2]; a20+=f[3*2 +0]; a21+=f[3*2 +1]; a22+=f[3*2 +2]; return *this; }
 template<typename U> mat3& operator -= ( const U * const f ){ a00-=f[3*0 +0]; a01-=f[3*0 +1]; a02-=f[3*0 +2]; a10-=f[3*1 +0]; a11-=f[3*1 +1]; a12-=f[3*1 +2]; a20-=f[3*2 +0]; a21-=f[3*2 +1]; a22-=f[3*2 +2]; return *this; }
 mat3& set( const T& m00, const T& m01, const T& m02, const T& m10, const T& m11, const T& m12, const T& m20, const T& m21, const T& m22 ){ a00=m00; a01=m01; a02=m02; a10=m10; a11=m11; a12=m12; a20=m20; a21=m21; a22=m22; return *this; }
 mat3& plus( const T& m00, const T& m01, const T& m02, const T& m10, const T& m11, const T& m12, const T& m20, const T& m21, const T& m22 ){ a00+=m00; a01+=m01; a02+=m02; a10+=m10; a11+=m11; a12+=m12; a20+=m20; a21+=m21; a22+=m22; return *this; }
 mat3& minus( const T& m00, const T& m01, const T& m02, const T& m10, const T& m11, const T& m12, const T& m20, const T& m21, const T& m22 ){ a00-=m00; a01-=m01; a02-=m02; a10-=m10; a11-=m11; a12-=m12; a20-=m20; a21-=m21; a22-=m22; return *this; }
 const T* operator[](int i)const{ return a[i]; }
 T* operator[](int i){ return a[i]; }
 const T* c_array()const{ return a[0]; }

 friend class mat3<double>;
 friend class mat3<float>;
 friend class mat3<long>;
 friend class mat3<int>;

 friend int operator==<T> (const mat3 & ml, const mat3 & mr);
 friend int operator!=<T> (const mat3 & ml, const mat3 & mr);
 friend vec3<T> operator *<T> (const mat3 & ml, const vec3<T>& vr);
 friend vec3<T> operator *<T> (const vec3<T>& vl, const mat3 & mr);
 friend mat3 operator *<T> (const mat3 & ml, const mat3 & mr);

};



template <typename T,typename U> inline mat3<T> operator + (const U & fl, const mat3<T>& mr){ return mat3<T>(mr)+=fl; }
template <typename T,typename U> inline mat3<T> operator + (const mat3<T>& ml, const U & fr){ return mat3<T>(ml)+=fr; }
template <typename T,typename U> inline mat3<T> operator - (const U & fl, const mat3<T>& mr){ return mat3<T>(fl)-=mr; }
template <typename T,typename U> inline mat3<T> operator - (const mat3<T>& ml, const U & fr){ return mat3<T>(ml)-=fr; }
template <typename T,typename U> inline vec3<T> operator * (const vec3<T>& vl, const U & fr){ return vec3<T>(vl)*=fr; }
template <typename T,typename U> inline vec3<T> operator * (const U & fl, const vec3<T>& vr){ return vec3<T>(vr)*=fl; }
template <typename T,typename U> inline mat3<T> operator * (const mat3<T>& ml, const U & fr){ return mat3<T>(ml)*=fr; }
template <typename T,typename U> inline mat3<T> operator * (const U & fl, const mat3<T>& mr){ return mat3<T>(mr)*=fl; }
template <typename T,typename U> inline vec3<T> operator / (const vec3<T>& vl, const U & fr){ return vec3<T>(vl)/=fr; }
template <typename T,typename U> inline mat3<T> operator / (const mat3<T>& ml, const U & fr){ return mat3<T>(ml)/=fr; }


template <typename T> inline vec3<T> operator + (const vec3<T>& v){ return v; }
template <typename T> inline vec3<T> operator - (const vec3<T>& v){ return vec3<T>(v)*=-1; }
template <typename T> inline mat3<T> operator + (const mat3<T>& m){ return m; }
template <typename T> inline mat3<T> operator - (const mat3<T>& m){ return mat3<T>(m)*=-1; }
template <typename T> inline vec3<T> operator + (const vec3<T>& vl, const vec3<T>& vr){ return vec3<T>(vl)+=vr; }
template <typename T> inline vec3<T> operator - (const vec3<T>& vl, const vec3<T>& vr){ return vec3<T>(vl)-=vr; }
template <typename T> inline mat3<T> operator + (const mat3<T>& ml, const mat3<T>& mr){ return mat3<T>(ml)+=mr; }
template <typename T> inline mat3<T> operator - (const mat3<T>& ml, const mat3<T>& mr){ return mat3<T>(ml)-=mr; }


template <typename T>
inline int operator== (const vec3<T>& vl, const vec3<T>& vr){ return (vl.b[0]==vr.b[0]&& vl.b[1]==vr.b[1]&& vl.b[2]==vr.b[2]); }
template <typename T>
inline int operator!= (const vec3<T>& vl, const vec3<T>& vr){ return (vl.b[0]!=vr.b[0]|| vl.b[1]!=vr.b[1]|| vl.b[2]!=vr.b[2]); }
template <typename T>
inline int operator < (const vec3<T>& vl, const vec3<T>& vr){ return (vl.b[0] <vr.b[0]&& vl.b[1] <vr.b[1]&& vl.b[2]!=vr.b[2]); }
template <typename T>
inline int operator<= (const vec3<T>& vl, const vec3<T>& vr){ return (vl.b[0]<=vr.b[0]&& vl.b[1]<=vr.b[1]&& vl.b[2]<=vr.b[2]); }
template <typename T>
inline int operator == (const mat3<T>& ml, const mat3<T>& mr){ return (ml.a[0][0]==mr.a[0][0]&& ml.a[0][1]==mr.a[0][1]&& ml.a[0][2]==mr.a[0][2]&& ml.a[1][0]==mr.a[1][0]&& ml.a[1][1]==mr.a[1][1]&& ml.a[1][2]==mr.a[1][2]&& ml.a[2][0]==mr.a[2][0]&& ml.a[2][1]==mr.a[2][1]&& ml.a[2][2]==mr.a[2][2]); }
template <typename T>
inline int operator != (const mat3<T>& ml, const mat3<T>& mr){ return (ml.a[0][0]!=mr.a[0][0]|| ml.a[0][1]!=mr.a[0][1]|| ml.a[0][2]!=mr.a[0][2]|| ml.a[1][0]!=mr.a[1][0]|| ml.a[1][1]!=mr.a[1][1]|| ml.a[1][2]!=mr.a[1][2]|| ml.a[2][0]!=mr.a[2][0]|| ml.a[2][1]!=mr.a[2][1]|| ml.a[2][2]!=mr.a[2][2]); }

template <typename T>
inline vec3<T> operator * (const mat3<T>& ml, const vec3<T>& vr){ return vec3<T>(ml.a[0][0]*vr.b[0]+ ml.a[0][1]*vr.b[1]+ ml.a[0][2]*vr.b[2], ml.a[1][0]*vr.b[0]+ ml.a[1][1]*vr.b[1]+ ml.a[1][2]*vr.b[2], ml.a[2][0]*vr.b[0]+ ml.a[2][1]*vr.b[1]+ ml.a[2][2]*vr.b[2]); }
template <typename T>
inline vec3<T> operator * (const vec3<T>& vl, const mat3<T>& mr){ return vec3<T>(vl.b[0]*mr.a[0][0]+ vl.b[1]*mr.a[1][0]+ vl.b[2]*mr.a[2][0], vl.b[0]*mr.a[0][1]+ vl.b[1]*mr.a[1][1]+ vl.b[2]*mr.a[2][1], vl.b[0]*mr.a[0][2]+ vl.b[1]*mr.a[1][2]+ vl.b[2]*mr.a[2][2]); }
template <typename T>
inline T operator * (const vec3<T>& vl, const vec3<T>& vr){ return vl.b[0]*vr.b[0]+ vl.b[1]*vr.b[1]+ vl.b[2]*vr.b[2]; }
template <typename T>
inline mat3<T> operator * (const mat3<T>& ml, const mat3<T>& mr){return mat3<T>(ml.a[0][0]*mr.a[0][0]+ ml.a[0][1]*mr.a[1][0]+ ml.a[0][2]*mr.a[2][0], ml.a[0][0]*mr.a[0][1]+ ml.a[0][1]*mr.a[1][1]+ ml.a[0][2]*mr.a[2][1], ml.a[0][0]*mr.a[0][2]+ ml.a[0][1]*mr.a[1][2]+ ml.a[0][2]*mr.a[2][2], ml.a[1][0]*mr.a[0][0]+ ml.a[1][1]*mr.a[1][0]+ ml.a[1][2]*mr.a[2][0], ml.a[1][0]*mr.a[0][1]+ ml.a[1][1]*mr.a[1][1]+ ml.a[1][2]*mr.a[2][1], ml.a[1][0]*mr.a[0][2]+ ml.a[1][1]*mr.a[1][2]+ ml.a[1][2]*mr.a[2][2], ml.a[2][0]*mr.a[0][0]+ ml.a[2][1]*mr.a[1][0]+ ml.a[2][2]*mr.a[2][0], ml.a[2][0]*mr.a[0][1]+ ml.a[2][1]*mr.a[1][1]+ ml.a[2][2]*mr.a[2][1], ml.a[2][0]*mr.a[0][2]+ ml.a[2][1]*mr.a[1][2]+ ml.a[2][2]*mr.a[2][2]);}
template <typename T>
inline mat3<T> operator % (const vec3<T>& vl, const vec3<T>& vr){ return mat3<T>( vl.b[0]*vl.b[0], vl.b[0]*vl.b[1], vl.b[0]*vl.b[2], vl.b[1]*vl.b[0], vl.b[1]*vl.b[1], vl.b[1]*vl.b[2], vl.b[2]*vl.b[0], vl.b[2]*vl.b[1], vl.b[2]*vl.b[2]);}

template <typename T>
inline vec3<T> cross(const vec3<T>& vl, const vec3<T>& vr){
 return vec3<T>(
  vl.b[1]*vr.b[2]-vl.b[2]*vr.b[1],
  vl.b[2]*vr.b[0]-vl.b[0]*vr.b[2],
  vl.b[0]*vr.b[1]-vl.b[1]*vr.b[0]
  );
}


#endif//__VEC3T_HPP__INCLUDED__

