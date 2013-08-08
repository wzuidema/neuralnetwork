layer input, combinedinputs, hidden, output, recurrent, target;

int prevtime=0;

double theinput[] = {1.0,0.0,0.0,0.0,0.0,0.0};
//double theoutput[] = {1.0,0.0,0.5};
double[] scaled;

void setup() 
{
  size(800, 700); 
//  noStroke();
  rectMode(CENTER);
  
  input = new layer(theinput.length,null);
  recurrent = new layer(5,null);
  combinedinputs = new layer(input,recurrent);
  hidden = new layer(5, combinedinputs);
  output = new layer(theinput.length, hidden);
  target = new layer(theinput.length, null);
  
  input.setCoordinates(150,200,50,0);
  recurrent.setCoordinates(500,250,50,0);
  hidden.setCoordinates(150,300,50,0);
  output.setCoordinates(150,400,50,0);
  target.setCoordinates(150,550,50,0);
  
  input.setActivations(theinput);
  hidden.computeActivations();
  output.computeActivations();
}

void draw()
{
  if (millis()>prevtime+500) {prevtime=millis(); trainNetwork(); }
  background(51); 
  input.display();
  recurrent.display();
  hidden.display();
  output.display();
  target.display();
}

void keyPressed() {
  int num=1;
  if (key=='q') {num=1000;}
  if (key=='r') {num=10000;}
    for (int g=0; g<num; g++) {
      trainNetwork();
    }
//  output.updateWeights();
}

void trainNetwork() {
  rotateArray(theinput);
  scaled = scalartimesvector(1.0,theinput); // random(1.0)
  input.setActivations(scaled);
  recurrent.copyActivations(hidden);
  hidden.computeActivations();
  output.computeActivations();
  rotateArray(scaled);
  target.setActivations(scaled);
  output.computeErrors(scaled);
  output.backpropagate();
  output.updateWeights();
  hidden.updateWeights();
}

void randomizeArray(double[] old) {
  int candidate;
  double oldval;
  for (int i = 0; i < old.length; i++) {
    candidate = int(i + random(old.length - i));
    oldval = old[i]; old[i] = old[candidate]; old[candidate] = oldval;
  }
}

void rotateArray(double[] old) {
  double oldval = old[0];
  for (int i = 0; i < old.length-1; i++)
    old[i] = old[i+1];
  old[old.length-1] = oldval;
}

double[] scalartimesvector(double scalar, double[] vector) {
  double[] newvec = new double[vector.length];
  for (int i = 0; i < vector.length; i++) 
   newvec[i] = vector[i] * scalar;
   return newvec;
}
  
