double eta = 0.1;

class layer {
  int size;
  layer donor;
  
  double weights[][], activations[], errors[];
  int x[], y[], diameter;
  boolean coordinatesSet=false;
  
 layer(int thesize, layer thedonor) {
   size = thesize;  
   donor = thedonor;

   errors = new double[size];
   activations = new double[size];
   for (int i=0; i<size; i++) 
     activations[i] = random(1.0);
   
   if (donor!=null) {
     weights = new double[size][];
     for (int i=0; i<size; i++) {
      weights[i] = new double[donor.size];
      for (int j=0; j<donor.size; j++) {
        weights[i][j] = random(1.0);
      }
     }
   }
 }
 
 void setCoordinates(int x0, int y0, int dx, int dy) {
   coordinatesSet=true;
    x = new int[size];
    y = new int[size];  
    for (int i=0; i<size; i++) {
      x[i] = x0 + i*dx;
      y[i] = y0 + i*dy;
    }
    diameter = int(0.8 * max(dx,dy));
 }
 
 void display() {
   if (coordinatesSet) {
     for (int i=0; i<size; i++) {
       strokeWeight(0);
       fill((int)(256*activations[i]),(int)(256*(1.0-activations[i])),0);
       rect(x[i],y[i],diameter,diameter);
       if (donor!=null) {
         for (int j=0; j<donor.size; j++) {
           if (weights[i][j]<0.0) stroke(256,256,256); else stroke(0,0,0);
           strokeWeight(min(8,abs((int)(weights[i][j]*4))));
           line(x[i],y[i],donor.x[j],donor.y[j]);
         }
       }
     }
   }
 }
 
 void setActivations(double theactivations[]) {
   if (size != theactivations.length) println("Mismatch!");
   for (int i=0; i<size; i++)
     activations[i] = theactivations[i];
 }
 
 void computeActivations() {
   if (donor!=null) {
     for (int i=0; i<size; i++) {
       activations[i] = 0.0;
       for (int j=0; j<donor.size; j++) {
         activations[i]+=weights[i][j]*donor.activations[j];
       }
       activations[i] = sigmoid(activations[i]);
     }
   }
 }

  void computeErrors(double target[]) {
   if (size!=target.length) {println("Mismatch! (computeErrors())");}
   for (int i=0; i<size; i++) {
     errors[i] = activations[i] - target[i];
     }
  }
  
  void backpropagate() {
   if (donor!=null) {
       for (int j=0; j<donor.size; j++) {
         donor.errors[j]=0.0;
         for (int i=0; i<size; i++) {
            donor.errors[j] += weights[i][j]*errors[i]*donor.activations[j];
         }
       }
       donor.backpropagate();
   }
  }
  
  void updateWeights() {
   if (donor!=null) {
       for (int j=0; j<donor.size; j++) {
         for (int i=0; i<size; i++) {
            weights[i][j]-=donor.activations[j]*errors[i]*eta;
         }
       }
   }
  }
}

double sigmoid(double x) {
  return 1.0 / (1.0 + exp((float)-x));
}
