double eta = 0.01;
int historyLength=500;

class node {
  double weights[], activation, error;
  int x, y, diameter;
  
  double history[];
  
  node(double theactivation) {
    activation = theactivation;
    history = new double[historyLength];
  }
  
  void setCoordinates(int thex,int they,int thediameter) {
    x=thex; y=they; diameter=thediameter;
  }
  
  void activation(double value) {
    activation = value;
    history[round%historyLength] = value;
  }
  
  void showHistory(int xorigin, int yorigin, int xscale, int yscale) {
    for (int h=1; h<historyLength; h++)
      line(xorigin+xscale*(h-1),(int)(yorigin+yscale*history[h-1]),xorigin+xscale*h,(int)(yorigin+yscale*history[h]));
  }
}

class layer {
  int size;
  layer donor;
  node nodes[];
  double summedSqError=0.0;
  
  boolean coordinatesSet=false;
 
 layer(int thesize, layer thedonor) {
   size = thesize;  
   donor = thedonor;

   nodes = new node[size];
   for (int i=0; i<size; i++) 
     nodes[i] = new node(random(1.0));
   
   if (donor!=null) {
     for (int i=0; i<size; i++) {
      nodes[i].weights = new double[donor.size];
      for (int j=0; j<donor.size; j++) {
        if (i==j) nodes[i].weights[j] = random(0.4)+0.6;
        else nodes[i].weights[j] = random(0.4)-0.2;
      }
     }
   }
 }

 layer(layer l1, layer l2) {
   size = l1.size + l2.size;  

   nodes = new node[size];
   for (int i=0; i<l1.size; i++) 
     nodes[i] = l1.nodes[i];
   for (int i=0; i<l2.size; i++) 
     nodes[l1.size+i] = l2.nodes[i];
   
   donor = null;
   if (l1.coordinatesSet && l2.coordinatesSet) coordinatesSet=true;
 } 
 
 void setCoordinates(int x0, int y0, int dx, int dy) {
   coordinatesSet=true;
    for (int i=0; i<size; i++) {
      nodes[i].x = x0 + i*dx;
      nodes[i].y = y0 + i*dy;
      nodes[i].diameter = int(0.8 * max(dx,dy));
    }
 }
 
 void display() {
   if (coordinatesSet) {
     for (int i=0; i<size; i++) {
       strokeWeight(0);
       fill((int)(256*nodes[i].activation),(int)(256*(1.0-nodes[i].activation)),0);
       rect(nodes[i].x,nodes[i].y,nodes[i].diameter,nodes[i].diameter);
       if (donor!=null && donor.coordinatesSet) {
         for (int j=0; j<donor.size; j++) {
           if (nodes[i].weights[j]<0.0) { 
             stroke(256,256,256);
             strokeWeight(max(1.0,(int)(log(-(float)nodes[i].weights[j]*4)))); } 
           else { 
             stroke(0,0,0);
             strokeWeight(max(1.0,(int)(log((float)nodes[i].weights[j]*4)))); }
           line(nodes[i].x,nodes[i].y,donor.nodes[j].x,donor.nodes[j].y);
         }
       }
       
       strokeWeight(2);
       stroke(256,256-(i*(256/size)),i*(256/size));
       nodes[i].showHistory(nodes[0].x-550,nodes[0].y-nodes[0].diameter/2,1,nodes[0].diameter);
     }
   }
 }
 
 void setActivations(double theactivations[]) {
   if (size != theactivations.length) println("Mismatch! (setActivations)");
   for (int i=0; i<size; i++)
     nodes[i].activation(theactivations[i]);
 }
 
 void copyActivations(layer source) {
   if (size != source.size) println("Mismatch! (copyActivations)");
   for (int i=0; i<size; i++)
     nodes[i].activation(source.nodes[i].activation);
   
 }
 
 void computeActivations() {
   if (donor!=null) {
     for (int i=0; i<size; i++) {
       double activation = 0.0;
       for (int j=0; j<donor.size; j++) {
         activation+=nodes[i].weights[j]*donor.nodes[j].activation;
       }
       nodes[i].activation(sigmoid(activation));
     }
   }
 }

double[] average(int fromTime, int toTime) {
  double[] average = new double[size];
  for (int i=0; i<size; i++) {
    average[i] = 0.0;
    for (int t=fromTime; t<toTime; t++) 
      average[i] += nodes[i].history[(500+round+t)%historyLength];
    average[i]/=(toTime-fromTime);
  }
  return average;
}

  int sign(double value) {
   if (value<0.0) return -1;
   else return 1; 
  }

  void computeErrors(double target[]) {
   if (size!=target.length) {println("Mismatch! (computeErrors())");}
   summedSqError *= 100.0;
   for (int i=0; i<size; i++) {
     float error = (float)nodes[i].activation - (float)target[i];
     nodes[i].error = sign(error) * pow(error,2.0);
     summedSqError += error*error;
   } 
   summedSqError /= 101.0;
  }
  
  void computeErrors(layer target) {
   if (size!=target.size) {println("Mismatch! (computeErrors())");}
   summedSqError *= 100.0;
   for (int i=0; i<size; i++) {
     float error = (float)nodes[i].activation - (float)target.nodes[i].activation;
     nodes[i].error = sign(error) * pow(error,2.0);
     summedSqError += error*error;
   } 
   summedSqError /= 101.0;
  }
  
  void backpropagate() {
   if (donor!=null) {
       for (int j=0; j<donor.size; j++) {
         donor.nodes[j].error=0.0;
         for (int i=0; i<size; i++) {
            donor.nodes[j].error += nodes[i].weights[j]*nodes[i].error*donor.nodes[j].activation;
         }
       }
       donor.backpropagate();
   }
  }
  
  void updateWeights() {
   if (donor!=null) {
       for (int j=0; j<donor.size; j++) {
         for (int i=0; i<size; i++) {
            nodes[i].weights[j]-=donor.nodes[j].activation*nodes[i].error*eta;
         }
       }
   }
  }
}

double sigmoid(double x) {
  return 1.0 / (1.0 + exp(-(float)x));
}
