class autoencoder {
 layer target, previous, hidden, recurrent, output, input; 
 boolean coordinatesSet;
 int size, timescale;
 int xcoor, ycoor;
 
 autoencoder(int thesize,layer thetarget, int thetimescale) {
   target=thetarget;
   size=thesize;
   timescale=thetimescale;
   previous = new layer(target.size,null);
   recurrent = new layer(size,null);
   input = new layer(previous,recurrent);
   hidden = new layer(size,input);
   output = new layer(target.size,hidden);
 }
 
 void setCoordinates(int x0, int y0, int dx, int dy) {
   coordinatesSet=true;
   hidden.setCoordinates(x0, y0, dx, 0);
   output.setCoordinates(x0, y0+dy, dx, 0);
   xcoor=x0-dx;
   ycoor=y0+dy;
 }
 
 void display() {
   if (coordinatesSet) {
     hidden.display();
     output.display(); }
   text(String.format("%.2f", output.summedSqError/output.size),xcoor,ycoor);
 }

 void train() {
  previous.setActivations(target.average(-2*timescale,-1*timescale));
  recurrent.copyActivations(hidden);
  hidden.computeActivations();
  output.computeActivations();
  output.computeErrors(target.average(-1*timescale,1));
  output.backpropagate();
  output.updateWeights();
  hidden.updateWeights();
}

}
