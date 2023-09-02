// Gmsh project created on Sat Jul 22 02:48:00 2023
SetFactory("OpenCASCADE");

//******************************Variaveis para desenvolvimento da malha********************************


Hesq = 20 ; PHesq = 1 ;

Hdir = 20 ; PHdir = 1.0 ;

V = 20 ; PV =1 ;

//******************************************* PONTOS*************************************
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

Point(26) = {0, 0.75, 0, 1.0};


Point(27) = {0.25, 0.75, 0, 1.0};
Point(28) = {0.05, 0.75, -0, 1.0};
Point(29) = {0.25, 0.95, -0, 1.0};
Point(30) = {0.05, 0.25, -0, 1.0};
Point(31) = {0.25, 0.05, -0, 1.0};//+

//Linhas
Line(1) = {4, 10};
//+
Line(2) = {10, 20};
//+
Line(3) = {20, 17};
//+
Line(4) = {17, 3};
//+
Line(5) = {3, 6};
//+
Line(6) = {6, 7};
//+
Line(7) = {7, 2};
//+
Line(8) = {2, 16};
//+
Line(9) = {16, 12};
//+
Line(10) = {12, 19};
//+
Line(11) = {19, 15};
//+
Line(12) = {15, 11};
//+
Line(13) = {11, 1};
//+
Line(14) = {1, 9};
//+
Line(15) = {9, 25};
//+
Line(16) = {25, 26};
//+
Line(17) = {26, 5};
//+
Line(18) = {5, 4};
//+
Line(19) = {21, 10};
//+
Line(20) = {21, 5};
//+
Line(21) = {29, 27};
//+
Line(22) = {27, 28};
//+
Line(23) = {18, 30};
//+
Line(24) = {18, 31};
//+
Line(25) = {31, 22};
//+
Line(26) = {22, 30};
//+
Line(27) = {30, 25};
//+
Line(28) = {31, 19};
//+
Line(29) = {22, 11};
//+
Line(30) = {22, 9};
//+
Line(31) = {18, 27};
//+
Line(32) = {30, 28};
//+
Line(33) = {28, 26};
//+
Line(34) = {29, 20};
//+
Line(35) = {29, 21};
//+
Line(36) = {21, 28};
//+
Line(37) = {24, 6};
//+
Line(38) = {24, 29};
//+
Line(39) = {17, 24};
//+
Line(40) = {7, 23};
//+
Line(41) = {23, 16};
//+
Line(42) = {23, 31};

//Criando os planos
//+
Curve Loop(1) = {23, -26, -25, -24};
//+
Plane Surface(1) = {1};
//+
Curve Loop(2) = {42, 28, -10, -9, -41};
//+
Plane Surface(2) = {2};
//+
Curve Loop(3) = {28, 11, 12, -29, -25};
//+
Plane Surface(3) = {3};
//+
Curve Loop(4) = {38, 34, 3, 39};
//+
Plane Surface(4) = {4};
//+
Curve Loop(5) = {34, -2, -19, -35};
//+
Plane Surface(5) = {5};
//+
Curve Loop(6) = {21, 22, -36, -35};
//+
Plane Surface(6) = {6};
//+
Curve Loop(7) = {31, 22, -32, -23};
//+
Plane Surface(7) = {7};
//+
//Curve Loop(8) = {31, -21, -38, 37, 6, 40, 42, -24};
//+
//Plane Surface(8) = {8};
//+
Curve Loop(9) = {36, 33, 17, -20};
//+
Plane Surface(9) = {9};
//+
Curve Loop(10) = {32, 33, -16, -27};
//+
Plane Surface(10) = {10};
//+
Curve Loop(11) = {26, 27, -15, -30};
//+
Plane Surface(11) = {11};
//+
Curve Loop(12) = {40, 41, -8, -7};
//+
Plane Surface(12) = {12};
//+
Curve Loop(13) = {4, 5, -37, -39};
//+
Plane Surface(13) = {13};
//+
Curve Loop(14) = {1, -19, 20, 18};
//+
Plane Surface(14) = {14};
//+
Curve Loop(15) = {30, -14, -13, -29};
//+
Plane Surface(15) = {15};



//Criando as malhas//+
//Verticais da esquerda para a direita , cima para baixo

Transfinite Curve {18} = V Using Progression PV;
//+
Transfinite Curve {19} = V Using Progression PV;
//+
Transfinite Curve {34} = V Using Progression PV;
//+
Transfinite Curve {39} = V Using Progression PV;
//+
Transfinite Curve {5} = V Using Progression PV;
//+
Transfinite Curve {17} = V Using Progression PV;
//+
Transfinite Curve {36} = V Using Progression PV;
//+
Transfinite Curve {21} = V Using Progression PV;
//+
Transfinite Curve {6} = V Using Progression PV;
//+
Transfinite Curve {16} = V Using Progression PV;
//+
Transfinite Curve {32} = V Using Progression PV;
//+
Transfinite Curve {31} = V Using Progression PV;
//+
Transfinite Curve {15} = V Using Progression PV;
//+
Transfinite Curve {26} = V Using Progression PV;
//+
Transfinite Curve {24} = V Using Progression PV;
//+
Transfinite Curve {14} = V Using Progression PV;
//+
Transfinite Curve {29} = V Using Progression PV;
//+
Transfinite Curve {28} = V Using Progression PV;
//+
Transfinite Curve {41} = V Using Progression PV;
//+
Transfinite Curve {7} = V Using Progression PV;

//Horizontais da esquerda para direita, de cima para baixo

//+
Transfinite Curve {1} = Hesq Using Progression PHesq;
//+
Transfinite Curve {2} = Hesq Using Progression PHesq;
//+
Transfinite Curve {3} = Hesq Using Progression PHesq;
//+
Transfinite Curve {4} = Hesq Using Progression PHesq;
//+
Transfinite Curve {20} = Hesq Using Progression PHesq;
//+
Transfinite Curve {35} = Hesq Using Progression PHesq;
//+
Transfinite Curve {38} = Hesq Using Progression PHesq;
//+
Transfinite Curve {37} = Hesq Using Progression PHesq;
//+
Transfinite Curve {33} = Hesq Using Progression PHesq;
//+
Transfinite Curve {22} = Hesq Using Progression PHesq;
//+
Transfinite Curve {27} = Hesq Using Progression PHesq;
//+
Transfinite Curve {23} = Hesq Using Progression PHesq;
//+
Transfinite Curve {30} = Hesq Using Progression PHesq;
//+
Transfinite Curve {25} = Hesq Using Progression PHesq;
//+
Transfinite Curve {42} = Hesq Using Progression PHesq;
//+
Transfinite Curve {40} = Hesq Using Progression PHesq;
//+
Transfinite Curve {13} = Hesq Using Progression PHesq;
//+
Transfinite Curve {11} = Hesq Using Progression PHesq;
//+
Transfinite Curve {12} = Hesq Using Progression PHesq;
//+
Transfinite Curve {10} = Hesq Using Progression PHesq;
//+
Transfinite Curve {9} = Hesq Using Progression PHesq;
//+
Transfinite Curve {8} = Hesq Using Progression PHesq;


//Planos esquerda para direita, cima p baixo
//+
Transfinite Surface {14};
//+
Transfinite Surface {5};
//+
Transfinite Surface {4};
//+
Transfinite Surface {13};
//+
Transfinite Surface {9};
//+
Transfinite Surface {6};
//+
Transfinite Surface {8};
//+
Transfinite Surface {10};
//+
Transfinite Surface {7};
//+
Transfinite Surface {11};
//+
Transfinite Surface {1};
//+
Transfinite Surface {15};
//+
Transfinite Surface {3};
//+
Transfinite Surface {2};
//+
Transfinite Surface {12};
//+
Recursive Delete {
  Point{8}; 
}
//+
Recombine Surface {14};
//+
Recombine Surface {5};
//+
Recombine Surface {4};
//+
Recombine Surface {13};
//+
Recombine Surface {9};
//+
Recombine Surface {6};
//+
Recombine Surface {8};
//+
Recombine Surface {10};
//+
Recombine Surface {7};
//+
Recombine Surface {11};
//+
Recombine Surface {1};
//+
Recombine Surface {15};
//+
Recombine Surface {3};
//+
Recombine Surface {2};
//+
Recombine Surface {12};

//+
Point(32) = {2, 0.75, 0, 1.0};
//+
Point(33) = {2, 0.25, 0, 1.0};
//+
Point(34) = {1.95, 0.75, 0, 1.0};
//+
Point(35) = {1.95, 0.25, 0, 1.0};
//+
Line(43) = {34, 24};
//+
Line(44) = {34, 35};
//+
Line(45) = {23, 35};
//+
Line(46) = {34, 27};
//+
Line(47) = {18, 35};
//+
Curve Loop(16) = {46, -21, -38, -43};
//+
Plane Surface(16) = {16};
//+
Curve Loop(17) = {31, -46, 44, -47};
//+
Plane Surface(17) = {17};
//+
Curve Loop(18) = {42, -24, 47, -45};
//+
Plane Surface(18) = {18};
//+
Line(48) = {33, 35};
//+
Line(49) = {32, 34};
//+
Curve Loop(19) = {6, 40, 45, -44, 43, 37};
//+
Plane Surface(19) = {19};
//+
Recursive Delete {
  Surface{2}; Surface{3}; Surface{12}; Surface{15}; 
}
//+
Recursive Delete {
  Point{13}; 
}
//+
Recursive Delete {
  Point{14}; 
}
//+
Transfinite Curve {46} = V Using Progression PV;
//+
Transfinite Curve {47} = V Using Progression PV;
//+
Transfinite Curve {43} = V Using Progression PV;
//+
Transfinite Curve {44} = V Using Progression PV;
//+
Transfinite Curve {45} = V Using Progression PV;
//+
Transfinite Curve {48} = V Using Progression PV;
//+
Transfinite Curve {49} = V Using Progression PV;
//+
Transfinite Surface {17};
//+
Transfinite Surface {16};
//+
Transfinite Surface {18};
//+
Transfinite Surface {19};
//+
Recursive Delete {
  Surface{19}; 
}
//+
Recursive Delete {
  Surface{13}; 
}
//+
Recursive Delete {
  Point{33}; 
}
//+
Recursive Delete {
  Point{33}; 
}
//+
Recursive Delete {
  Curve{48}; 
}
//+
Recursive Delete {
  Curve{49}; 
}

//**************COND DE CONTORNO*****************8
//+
Physical Curve("eletrodo_top", 48) = {1, 2, 3};
//+
Physical Curve("eletrodo_low", 49) = {42, 25, 30};
//+
Physical Curve("esquerda", 50) = {18, 17, 16, 15};
//+
Physical Curve("direita", 51) = {39, 43, 44, 45};
//+
Physical Surface("dominio", 52) = {6, 5, 14, 9, 10, 7, 11, 1, 18, 17, 16, 4};
