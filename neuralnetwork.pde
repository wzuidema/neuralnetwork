PFont font12;
layer input;
autoencoder[] raams;

int prevtime=0, round=0;

double theinput[] = {1.0,0.0,0.0,0.0,0.0,0.0};
//double theoutput[] = {1.0,0.0,0.5};
double[] scaled;

void setup() 
{
  size(1000, 700); 
  rectMode(CENTER);
  
  font12 = createFont("Arial-Black",12);
  textFont(font12); 

  input = new layer(theinput.length,null);
  input.setCoordinates(600,100,50,0);
  input.setActivations(theinput);

  raams = new autoencoder[5];
  raams[0] = new autoencoder(5,input,1);
  raams[0].setCoordinates(600,200,50,-50);
  
  for (int r=1; r<5; r++) {
    raams[r] = new autoencoder(5-r,raams[r-1].hidden,(int)pow(2,r));
    raams[r].setCoordinates(600,200+r*100,50,-50);
  }
}

void draw()
{
  background(51);
  //if (millis()>prevtime+1000) {prevtime=millis(); for (int g=0; g<10000; g++) trainNetwork(); }
  
  text(""+round,900,650);
   
  input.display();
  for (int r=0; r<5; r++)
    raams[r].display();

//    stroke(256,256,256);
//  for (int t=0; t<50; t++)
//    line(t*10,350+sin(t)*10,10*(t+1),350+10*sin(t+1));
}

void keyPressed() {
  int num=1;
  if (key=='q') {num=1000;}
  if (key=='w') {num=100000;}
  if (key=='e') {num=1000000;}
  if (key=='r') {num=10000000;}
    for (int g=0; g<num; g++) {
      trainNetwork();
    }
//  output.updateWeights();
}

float nearesthalf(float value) {
  if (value>.75) return 1.0;
  else if (value<.25) return 0.0;
  else return 0.5;
}

void trainNetwork() {
  round++;
  
  float longwave = sin((float)round/64.0);
  float shortwave = sin((float)round/8.0);
  float wave = shortwave;
  
  if (longwave>0.0) {
  if (wave<-0.5) theinput[0]=1.0; else theinput[0]=0.0;
  if (wave>-0.5 && wave<0.5) theinput[1]=1.0; else theinput[1]=0.0;
  if (wave>0.5) theinput[2]=1.0; else theinput[2]=0.0;
  theinput[3]=0.0;
  theinput[4]=0.0;
  theinput[5]=0.0; }
  
  if (longwave<0.0) {
  if (wave<-0.5) theinput[3]=1.0; else theinput[3]=0.0;
  if (wave>-0.5 && wave<0.5) theinput[4]=1.0; else theinput[4]=0.0;
  if (wave>0.5) theinput[5]=1.0; else theinput[5]=0.0;
  theinput[0]=0.0;
  theinput[1]=0.0;
  theinput[2]=0.0; }
  
//  rotateArray(theinput);
//  scaled = scalartimesvector(1.0,theinput); // random(1.0)
  input.setActivations(theinput);
  
  for (int r=0; r<5; r++)
    if (round%(pow(2,r))==0) raams[r].train();
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
  
