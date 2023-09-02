// Gmsh project created on Thu Jul 13 12:38:59 2023
SetFactory("OpenCASCADE");
//+
Point(1) = {-0, 0, 0, 1.0};
//+
Point(2) = {2, 0, 0, 1.0};
//+
Point(3) = {2, 1, 0, 1.0};
//+
Point(4) = {0, 1, 0, 1.0};
//+
Point(5) = {0, 0.95, 0, 1.0};
//+
Point(6) = {2, 0.95, 0, 1.0};
//+
Point(7) = {2, 0.05, 0, 1.0};
//+
Point(8) = {0.6, 0.4, 0, 1.0};
//+
Point(9) = {0, 0.05, 0, 1.0};
//+
Recursive Delete {
  Point{8}; 
}

//+
Point(10) = {005, 1, 0, 1.0};
//+
Recursive Delete {
  Point{10}; 
}
//+
Point(10) = {0.05, 1, -0, 1.0};
//+
Point(11) = {0.05, 0, -0, 1.0};
//+
Point(12) = {0.95, 0, -0, 1.0};
//+
Point(13) = {0.95, 0, -0, 1.0};
//+
Point(14) = {0.95, 0, -0, 1.0};
//+
Point(15) = {0.195, 0, -0, 1.0};
//+
Point(16) = {1.95, 0, -0, 1.0};
//+
Point(17) = {1.95, 1, 0, 1.0};
//+
Point(18) = {0.25, 0.25, 0, 1.0};
//+
Point(19) = {0.25, 0, 0, 1.0};
//+
Point(20) = {0.25, 1, 0, 1.0};
//+
Point(21) = {0.05, 0.95, 0, 1.0};
//+
Point(22) = {0.05, 0.05, 0, 1.0};
//+
Point(23) = {1.95, 0.05, 0, 1.0};
//+
Point(24) = {1.95, 0.95, 0, 1.0};
//+
Point(25) = {0, 0.25, 0, 1.0};
//+
Point(26) = {0, 0.75, 0, 1.0};
//+
Point(27) = {0.25, 0.75, 0, 1.0};

//+
Recursive Delete {
  Point{15}; 
}
//+
Point(28) = {0.05, 0.75, -0, 1.0};
//+
Point(29) = {0.25, 0.95, -0, 1.0};
//+
Point(30) = {0.05, 0.25, -0, 1.0};


Point(31) = {0.25, 0.05, -0, 1.0};

//+
Line(1) = {9, 22};
//+
Line(2) = {22, 11};
//+
Line(3) = {11, 1};
//+
Line(4) = {1, 9};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Curve Loop(2) = {1, 2, 3, 4};
//+
Curve Loop(3) = {1, 2, 3, 4};
//+
Surface(1) = {3};
//+
Line(5) = {19, 31};
//+
Line(6) = {22, 31};
//+
Line(7) = {19, 11};
//+
Curve Loop(5) = {6, -5, 7, -2};
//+
Surface(2) = {5};
//+
Line(8) = {22, 30};
//+
Line(9) = {25, 30};
//+
Line(10) = {25, 9};
//+
Line(11) = {28, 26};
//+
Line(12) = {5, 26};
//+
Line(13) = {28, 21};
//+
Line(14) = {21, 5};
//+
Curve Loop(7) = {12, -11, 13, 14};
//+
Surface(3) = {7};
//+
Curve Loop(9) = {8, -9, 10, 1};
//+
Surface(4) = {9};
//+
Line(15) = {27, 28};
//+
Line(16) = {29, 27};
//+
Line(17) = {29, 21};
//+
Curve Loop(11) = {16, 15, 13, -17};
//+
Surface(5) = {11};
//+
Line(18) = {31, 18};
//+
Line(19) = {18, 30};
//+
Curve Loop(13) = {19, -8, 6, 18};
//+
Surface(6) = {13};
//+
Line(20) = {4, 5};
//+
Line(21) = {21, 10};
//+
Line(22) = {10, 4};
//+
Curve Loop(15) = {21, 22, 20, -14};
//+
Surface(7) = {15};
//+
Line(23) = {20, 29};
//+
Line(24) = {20, 10};
//+
Curve Loop(17) = {17, 21, -24, 23};
//+
Surface(8) = {17};
//+
Line(25) = {20, 17};
//+
Line(26) = {29, 24};
//+
Line(27) = {17, 24};
//+
Line(28) = {23, 7};
//+
Line(29) = {16, 23};
//+
Line(30) = {16, 2};
//+
Line(31) = {2, 7};
//+
Curve Loop(19) = {29, 28, -31, -30};
//+
Surface(9) = {19};
//+
Curve Loop(21) = {26, -27, -25, 23};
//+
Surface(10) = {21};
//+
Line(32) = {19, 16};
//+
Line(33) = {31, 23};
//+
Curve Loop(23) = {33, -29, -32, 5};
//+
Surface(11) = {23};
//+
Line(34) = {6, 3};
//+
Line(35) = {3, 17};
//+
Line(36) = {6, 24};
//+
Curve Loop(25) = {34, 35, 27, -36};
//+
Surface(12) = {25};
//+
Line(37) = {24, 23};
//+
Line(38) = {6, 7};
//+
Curve Loop(27) = {37, 28, -38, 36};
//+
Surface(13) = {27};
//+
Line(39) = {28, 30};
//+
Line(40) = {25, 26};
//+
Curve Loop(29) = {39, -9, 40, -11};
//+
Surface(14) = {29};
//+
Line(41) = {27, 18};
//+
Curve Loop(31) = {41, 19, -39, -15};
//+
Surface(15) = {31};
//+
Curve Loop(33) = {41, -18, 33, -37, -26, 16};
//+
Surface(16) = {33};
//+
Physical Surface("cantos_esquerdos", 42) = {7, 1};
//+
Physical Surface("lateral_esquerda_centro", 43) = {14};
//+
Physical Surface("lateral_esquerda_entre_canto_centro", 44) = {3, 4};
//+
Physical Surface("lado_esquerdo_meio", 45) = {15};
//+
Physical Surface("superior_inferior_menor", 46) = {8, 2};
//+
Physical Surface("meio", 47) = {16};
//+
Physical Surface("superior_inferior", 48) = {10, 11};
//+
Physical Surface("cantos_direito", 49) = {12, 9};
//+
Physical Surface("direita", 50) = {13};
//+
Physical Surface("quadrados_esquerdo", 51) = {5, 6};
//+

//+
Physical Surface("meio ", 52) = {16};

//+
Physical Surface("lado esquedo", 53) = {5, 6};
//+
Physical Surface("lado esquedo", 53) += {6};
