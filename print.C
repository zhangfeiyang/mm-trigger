#include "rootheader.h"
int get_ID(double x,double y,double z){

    double r = sqrt(x*x+y*y+z*z);
    double constheta = z/r;
    double phi;
    double pi = 3.141592654;
    if(y>0)
        phi = acos(x/sqrt(x*x+y*y));
    else
        phi = 2*pi - acos(x/sqrt(x*x+y*y));

    int line = (constheta + 1)/(2./12);
    int column = phi/(2*pi/12);
    int id = line*12 + column;

    return id;
}
void print(){

    ifstream fin("Map");
    double tmp,x,y,z;
    int pmtID[17739];
    TFile *f = new TFile("tmp.root","recreate");

    TNtuple *t = new TNtuple("t","","id");
    int index[17739];
    float X[17739],Y[17739],Z[17739];
    int id[17739];

    for(int i=0;i<17739;i++){
        fin>>pmtID[i]>>X[i]>>Y[i]>>Z[i];
	id[i] = get_ID(X[i],Y[i],Z[i]);
	t->Fill(id[i]);
    }

    TMath::Sort(17739,id,index,kFALSE);

    for(int i=0;i<17739;i++){
    	//cout << X[index[i]] << "\t" << Y[index[i]] << "\t" << Z[index[i]] << "\t"<< id[index[i]]<<"\n";
    	cout << pmtID[index[i]]<<"\n";
    }

    t->Write();
    f->Close();

}
