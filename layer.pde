double eta = 0.1;

class node {
  double weights[], activation, error;
  int x, y, diameter;
  
  node(double theactivation) {
    activation = theactivation; 
  }
  
  void setCoordinates(int thex,int they,int thediameter) {
    x=thex; y=they; diameter=thediameter;
  }
}

class layer {
  int size;
  layer donor;
  node nodes[];
  
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
        nodes[i].weights[j] = random(1.0);
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
       if (donor!=null) {
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
     }
   }
 }
 
 void setActivations(double theactivations[]) {
   if (size != theactivations.length) println("Mismatch! (setActivations)");
   for (int i=0; i<size; i++)
     nodes[i].activation = theactivations[i];
 }
 
 void copyActivations(layer source) {
   if (size != source.size) println("Mismatch! (copyActivations)");
   for (int i=0; i<size; i++)
     nodes[i].activation = source.nodes[i].activation;
   
 }
 
 void computeActivations() {
   if (donor!=null) {
     for (int i=0; i<size; i++) {
       nodes[i].activation = 0.0;
       for (int j=0; j<donor.size; j++) {
         nodes[i].activation+=nodes[i].weights[j]*donor.nodes[j].activation;
       }
       nodes[i].activation = sigmoid(nodes[i].activation);
     }
   }
 }

  void computeErrors(double target[]) {
   if (size!=target.length) {println("Mismatch! (computeErrors())");}
   for (int i=0; i<size; i++) {
     nodes[i].error = nodes[i].activation - target[i];
     }
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
