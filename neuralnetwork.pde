layer input, hidden, output, recurrent;

double theinput[] = {1.0,0.0,0.0,0.0,0.0,0.0};
//double theoutput[] = {1.0,0.0,0.5};
double[] scaled;

void setup() 
{
  size(800, 700); 
//  noStroke();
  rectMode(CENTER);
  
  input = new layer(theinput.length,null);
  hidden = new layer(3, input);
  output = new layer(theinput.length, hidden);
//  recurrent = new layer(2, hidden);
  
  input.setCoordinates(150,200,100,0);
  hidden.setCoordinates(150,400,100,0);
  output.setCoordinates(150,600,100,0);
//  recurrent.setCoordinates(500,500,50,0);
  
  input.setActivations(theinput);
  hidden.computeActivations();
  output.computeActivations();
}

void draw()
{
  background(51); 
  input.display();
  hidden.display();
  output.display();
  //recurrent.display();
}

void keyPressed() {
  int num=1;
  if (key=='q') {num=1000;}
  if (key=='r') {num=10000;}
    for (int g=0; g<num; g++) {
    randomizeArray(theinput);
    scaled = scalartimesvector(random(1.0),theinput); // random(1.0)
    input.setActivations(scaled);
  hidden.computeActivations();
  output.computeActivations();
  output.computeErrors(scaled);
  output.backpropagate();
  output.updateWeights();
  hidden.updateWeights();
    }
//  output.updateWeights();
}

void randomizeArray(double[] old) {
  int candidate;
  double oldval;
  for (int i = 0; i < old.length; i++) {
    candidate = int(i + random(old.length - i));
    oldval = old[i]; old[i] = old[candidate]; old[candidate] = oldval;
  }
}

double[] scalartimesvector(double scalar, double[] vector) {
  double[] newvec = new double[vector.length];
  for (int i = 0; i < vector.length; i++) 
   newvec[i] = vector[i] * scalar;
   return newvec;
}
  
