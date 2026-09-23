const RECORDED_RESULTS = {
  "controlled": [
    {
      "method": "Temporal",
      "macro_f1": 0.9283286734898359,
      "malicious_recall": 0.9872,
      "false_positive_rate": 0.13005,
      "C": 0,
      "D": 0
    },
    {
      "method": "Fixed",
      "macro_f1": 0.9289716516732478,
      "malicious_recall": 0.9859,
      "false_positive_rate": 0.1275,
      "C": 131,
      "D": 106
    },
    {
      "method": "Fixed-Dev16",
      "macro_f1": 0.9283420313127257,
      "malicious_recall": 0.9886,
      "false_positive_rate": 0.1314,
      "C": 53,
      "D": 52
    },
    {
      "method": "Logistic-0.5",
      "macro_f1": 0.9300667683014583,
      "malicious_recall": 0.98805,
      "false_positive_rate": 0.12745,
      "C": 107,
      "D": 38
    },
    {
      "method": "HGB-0.5",
      "macro_f1": 0.9275456444564673,
      "malicious_recall": 0.98705,
      "false_positive_rate": 0.13145,
      "C": 40,
      "D": 71
    },
    {
      "method": "Logistic-Dev16",
      "macro_f1": 0.928243395512087,
      "malicious_recall": 0.9883,
      "false_positive_rate": 0.1313,
      "C": 30,
      "D": 33
    },
    {
      "method": "Logistic-Global4",
      "macro_f1": 0.9283193095563352,
      "malicious_recall": 0.9883,
      "false_positive_rate": 0.13115,
      "C": 30,
      "D": 30
    },
    {
      "method": "HGB-Dev16",
      "macro_f1": 0.9284175177666676,
      "malicious_recall": 0.98865,
      "false_positive_rate": 0.1313,
      "C": 44,
      "D": 40
    },
    {
      "method": "HGB-Global4",
      "macro_f1": 0.9284934337251401,
      "malicious_recall": 0.98865,
      "false_positive_rate": 0.13115,
      "C": 44,
      "D": 37
    }
  ],
  "backend": [
    {
      "method": "Temporal",
      "macro_f1": 0.9283286734898359,
      "malicious_recall": 0.9872,
      "false_positive_rate": 0.13005,
      "C": 0,
      "D": 0
    },
    {
      "method": "Stats-ExtraTrees",
      "macro_f1": 0.7990323636209793,
      "malicious_recall": 0.9922,
      "false_positive_rate": 0.3802,
      "C": 971,
      "D": 5874
    },
    {
      "method": "Fixed",
      "macro_f1": 0.9340591381317742,
      "malicious_recall": 0.98805,
      "false_positive_rate": 0.11955,
      "C": 263,
      "D": 36
    },
    {
      "method": "Fixed-Dev16",
      "macro_f1": 0.9340587821728674,
      "malicious_recall": 0.9881,
      "false_positive_rate": 0.1196,
      "C": 262,
      "D": 35
    },
    {
      "method": "Logistic-0.5",
      "macro_f1": 0.9296193229321177,
      "malicious_recall": 0.9871,
      "false_positive_rate": 0.1274,
      "C": 83,
      "D": 32
    },
    {
      "method": "Logistic-Dev16",
      "macro_f1": 0.9276206934517228,
      "malicious_recall": 0.98715,
      "false_positive_rate": 0.1314,
      "C": 3,
      "D": 31
    },
    {
      "method": "Logistic-Global4",
      "macro_f1": 0.9276206934517228,
      "malicious_recall": 0.98715,
      "false_positive_rate": 0.1314,
      "C": 3,
      "D": 31
    },
    {
      "method": "HGB-0.5",
      "macro_f1": 0.9315398849533033,
      "malicious_recall": 0.98715,
      "false_positive_rate": 0.12365,
      "C": 161,
      "D": 34
    },
    {
      "method": "HGB-Dev16",
      "macro_f1": 0.9295430634711745,
      "malicious_recall": 0.98715,
      "false_positive_rate": 0.1276,
      "C": 82,
      "D": 34
    },
    {
      "method": "HGB-Global4",
      "macro_f1": 0.9295687537936431,
      "malicious_recall": 0.9871,
      "false_positive_rate": 0.1275,
      "C": 81,
      "D": 32
    }
  ],
  "raw": [
    {
      "method": "temporal",
      "macro_f1": 0.9720210280025658,
      "malicious_recall": 0.9954061006982727,
      "false_positive_rate": 0.05173821425229196,
      "C": 0,
      "D": 0
    },
    {
      "method": "strong_single",
      "macro_f1": 0.972105611181736,
      "malicious_recall": 0.9956183705280767,
      "false_positive_rate": 0.051783392010636135,
      "C": 91,
      "D": 38
    },
    {
      "method": "equal_average",
      "macro_f1": 0.9635463943315887,
      "malicious_recall": 0.989890253329785,
      "false_positive_rate": 0.06319400297527808,
      "C": 2434,
      "D": 7725
    },
    {
      "method": "fixed",
      "macro_f1": 0.9706710074688274,
      "malicious_recall": 0.9901500462558137,
      "false_positive_rate": 0.049121131251068935,
      "C": 2748,
      "D": 3596
    },
    {
      "method": "logistic_05",
      "macro_f1": 0.9736155188213669,
      "malicious_recall": 0.9957229213397711,
      "false_positive_rate": 0.04885006470100391,
      "C": 2371,
      "D": 1376
    },
    {
      "method": "hgb_05",
      "macro_f1": 0.9691368312369317,
      "malicious_recall": 0.9927289662776109,
      "false_positive_rate": 0.05482643673338991,
      "C": 689,
      "D": 2491
    },
    {
      "method": "logistic_dev",
      "macro_f1": 0.9713017769306125,
      "malicious_recall": 0.9958908362797653,
      "false_positive_rate": 0.05367763087835243,
      "C": 876,
      "D": 1324
    },
    {
      "method": "hgb_dev",
      "macro_f1": 0.971283345352373,
      "malicious_recall": 0.9962931984944683,
      "false_positive_rate": 0.0541229544963164,
      "C": 942,
      "D": 1401
    }
  ],
  "external": [
    {
      "method": "Temporal \u00b7 bundle 0",
      "macro_f1": 0.3742502075185957,
      "malicious_recall": 1.0,
      "false_positive_rate": 1.0,
      "C": 0,
      "D": 0
    },
    {
      "method": "Stats \u00b7 bundle 0",
      "macro_f1": 0.3743578841266089,
      "malicious_recall": 0.9998283888161424,
      "false_positive_rate": 0.9998648039657504,
      "C": 63,
      "D": 119
    },
    {
      "method": "Equal \u00b7 bundle 0",
      "macro_f1": 0.37421035359750593,
      "malicious_recall": 0.9998298309269311,
      "false_positive_rate": 1.0,
      "C": 0,
      "D": 118
    },
    {
      "method": "Fixed \u00b7 bundle 0",
      "macro_f1": 0.3742549034115614,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999957080624048,
      "C": 2,
      "D": 0
    },
    {
      "method": "Fixed-Dev16 \u00b7 bundle 0",
      "macro_f1": 0.3742549034115614,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999957080624048,
      "C": 2,
      "D": 0
    },
    {
      "method": "Logistic-0.5 \u00b7 bundle 0",
      "macro_f1": 0.3742549034115614,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999957080624048,
      "C": 2,
      "D": 0
    },
    {
      "method": "HGB-0.5 \u00b7 bundle 0",
      "macro_f1": 0.3742502075185957,
      "malicious_recall": 1.0,
      "false_positive_rate": 1.0,
      "C": 0,
      "D": 0
    },
    {
      "method": "Logistic-Dev16 \u00b7 bundle 0",
      "macro_f1": 0.3742549034115614,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999957080624048,
      "C": 2,
      "D": 0
    },
    {
      "method": "HGB-Dev16 \u00b7 bundle 0",
      "macro_f1": 0.3742502075185957,
      "malicious_recall": 1.0,
      "false_positive_rate": 1.0,
      "C": 0,
      "D": 0
    },
    {
      "method": "Logistic-Global4 \u00b7 bundle 0",
      "macro_f1": 0.3742549034115614,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999957080624048,
      "C": 2,
      "D": 0
    },
    {
      "method": "HGB-Global4 \u00b7 bundle 0",
      "macro_f1": 0.3742502075185957,
      "malicious_recall": 1.0,
      "false_positive_rate": 1.0,
      "C": 0,
      "D": 0
    },
    {
      "method": "Temporal \u00b7 bundle 1",
      "macro_f1": 0.22462893472057038,
      "malicious_recall": 0.377534509711174,
      "false_positive_rate": 0.9281100452799417,
      "C": 0,
      "D": 0
    },
    {
      "method": "Stats \u00b7 bundle 1",
      "macro_f1": 0.37831392748605336,
      "malicious_recall": 0.9998298309269311,
      "false_positive_rate": 0.9962359707289856,
      "C": 433380,
      "D": 33609
    },
    {
      "method": "Equal \u00b7 bundle 1",
      "macro_f1": 0.38756740741912304,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9876950149144832,
      "C": 432639,
      "D": 28770
    },
    {
      "method": "Fixed \u00b7 bundle 1",
      "macro_f1": 0.38532335880801727,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9897873344921565,
      "C": 431664,
      "D": 28770
    },
    {
      "method": "Fixed-Dev16 \u00b7 bundle 1",
      "macro_f1": 0.38525648614896274,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9898495675872873,
      "C": 431635,
      "D": 28770
    },
    {
      "method": "Logistic-0.5 \u00b7 bundle 1",
      "macro_f1": 0.38525966058718303,
      "malicious_recall": 0.9999942315568451,
      "false_positive_rate": 0.989845275649692,
      "C": 431633,
      "D": 28770
    },
    {
      "method": "HGB-0.5 \u00b7 bundle 1",
      "macro_f1": 0.38525966058718303,
      "malicious_recall": 0.9999942315568451,
      "false_positive_rate": 0.989845275649692,
      "C": 431633,
      "D": 28770
    },
    {
      "method": "Logistic-Dev16 \u00b7 bundle 1",
      "macro_f1": 0.3852550484803149,
      "malicious_recall": 0.9999942315568451,
      "false_positive_rate": 0.9898495675872873,
      "C": 431631,
      "D": 28770
    },
    {
      "method": "HGB-Dev16 \u00b7 bundle 1",
      "macro_f1": 0.3852550484803149,
      "malicious_recall": 0.9999942315568451,
      "false_positive_rate": 0.9898495675872873,
      "C": 431631,
      "D": 28770
    },
    {
      "method": "Logistic-Global4 \u00b7 bundle 1",
      "macro_f1": 0.22462893472057038,
      "malicious_recall": 0.377534509711174,
      "false_positive_rate": 0.9281100452799417,
      "C": 0,
      "D": 0
    },
    {
      "method": "HGB-Global4 \u00b7 bundle 1",
      "macro_f1": 0.22462893472057038,
      "malicious_recall": 0.377534509711174,
      "false_positive_rate": 0.9281100452799417,
      "C": 0,
      "D": 0
    },
    {
      "method": "Temporal \u00b7 bundle 2",
      "macro_f1": 0.37428307801402616,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999699564368334,
      "C": 0,
      "D": 0
    },
    {
      "method": "Stats \u00b7 bundle 2",
      "macro_f1": 0.3823345772818821,
      "malicious_recall": 0.9998298309269311,
      "false_positive_rate": 0.9925234447091139,
      "C": 3478,
      "D": 126
    },
    {
      "method": "Equal \u00b7 bundle 2",
      "macro_f1": 0.3742502581592682,
      "malicious_recall": 0.9998298309269311,
      "false_positive_rate": 0.9999635185304405,
      "C": 3,
      "D": 118
    },
    {
      "method": "Fixed \u00b7 bundle 2",
      "macro_f1": 0.3823832769950546,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.992517006802721,
      "C": 3473,
      "D": 0
    },
    {
      "method": "Fixed-Dev16 \u00b7 bundle 2",
      "macro_f1": 0.3823832769950546,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.992517006802721,
      "C": 3473,
      "D": 0
    },
    {
      "method": "Logistic-0.5 \u00b7 bundle 2",
      "macro_f1": 0.37428307801402616,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999699564368334,
      "C": 0,
      "D": 0
    },
    {
      "method": "HGB-0.5 \u00b7 bundle 2",
      "macro_f1": 0.3743582039575289,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999012854353098,
      "C": 32,
      "D": 0
    },
    {
      "method": "Logistic-Dev16 \u00b7 bundle 2",
      "macro_f1": 0.37428307801402616,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999699564368334,
      "C": 0,
      "D": 0
    },
    {
      "method": "HGB-Dev16 \u00b7 bundle 2",
      "macro_f1": 0.3794056949423889,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9952681388012619,
      "C": 2191,
      "D": 0
    },
    {
      "method": "Logistic-Global4 \u00b7 bundle 2",
      "macro_f1": 0.37428307801402616,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999699564368334,
      "C": 0,
      "D": 0
    },
    {
      "method": "HGB-Global4 \u00b7 bundle 2",
      "macro_f1": 0.3794056949423889,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9952681388012619,
      "C": 2191,
      "D": 0
    },
    {
      "method": "Temporal \u00b7 bundle 3",
      "macro_f1": 0.3742874021026267,
      "malicious_recall": 0.9999783683381692,
      "false_positive_rate": 0.999961372561643,
      "C": 0,
      "D": 0
    },
    {
      "method": "Stats \u00b7 bundle 3",
      "macro_f1": 0.3754016308756926,
      "malicious_recall": 0.9999985578892113,
      "false_positive_rate": 0.9989463293203716,
      "C": 506,
      "D": 19
    },
    {
      "method": "Equal \u00b7 bundle 3",
      "macro_f1": 0.3742595992685583,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999914161248096,
      "C": 19,
      "D": 18
    },
    {
      "method": "Fixed \u00b7 bundle 3",
      "macro_f1": 0.3742595992685583,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999914161248096,
      "C": 19,
      "D": 18
    },
    {
      "method": "Fixed-Dev16 \u00b7 bundle 3",
      "macro_f1": 0.3742595992685583,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999914161248096,
      "C": 19,
      "D": 18
    },
    {
      "method": "Logistic-0.5 \u00b7 bundle 3",
      "macro_f1": 0.3742549034115614,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999957080624048,
      "C": 17,
      "D": 18
    },
    {
      "method": "HGB-0.5 \u00b7 bundle 3",
      "macro_f1": 0.3742502075185957,
      "malicious_recall": 1.0,
      "false_positive_rate": 1.0,
      "C": 15,
      "D": 18
    },
    {
      "method": "Logistic-Dev16 \u00b7 bundle 3",
      "macro_f1": 0.3742572513445559,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999935620936071,
      "C": 18,
      "D": 18
    },
    {
      "method": "HGB-Dev16 \u00b7 bundle 3",
      "macro_f1": 0.3742502075185957,
      "malicious_recall": 1.0,
      "false_positive_rate": 1.0,
      "C": 15,
      "D": 18
    },
    {
      "method": "Logistic-Global4 \u00b7 bundle 3",
      "macro_f1": 0.3742572513445559,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9999935620936071,
      "C": 18,
      "D": 18
    },
    {
      "method": "HGB-Global4 \u00b7 bundle 3",
      "macro_f1": 0.3742502075185957,
      "malicious_recall": 1.0,
      "false_positive_rate": 1.0,
      "C": 15,
      "D": 18
    },
    {
      "method": "Temporal \u00b7 bundle 4",
      "macro_f1": 0.40554239377337675,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9706538766926329,
      "C": 0,
      "D": 0
    },
    {
      "method": "Stats \u00b7 bundle 4",
      "macro_f1": 0.37739205342899224,
      "malicious_recall": 0.999809641375889,
      "false_positive_rate": 0.9970793364664478,
      "C": 41,
      "D": 12487
    },
    {
      "method": "Equal \u00b7 bundle 4",
      "macro_f1": 0.3850686550064261,
      "malicious_recall": 0.9998384835916635,
      "false_positive_rate": 0.9899869095903345,
      "C": 0,
      "D": 9121
    },
    {
      "method": "Fixed \u00b7 bundle 4",
      "macro_f1": 0.40552901150340087,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9706667525054186,
      "C": 0,
      "D": 6
    },
    {
      "method": "Fixed-Dev16 \u00b7 bundle 4",
      "macro_f1": 0.40552901150340087,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9706667525054186,
      "C": 0,
      "D": 6
    },
    {
      "method": "Logistic-0.5 \u00b7 bundle 4",
      "macro_f1": 0.40554239377337675,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9706538766926329,
      "C": 0,
      "D": 0
    },
    {
      "method": "HGB-0.5 \u00b7 bundle 4",
      "macro_f1": 0.4055379330496064,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9706581686302281,
      "C": 0,
      "D": 2
    },
    {
      "method": "Logistic-Dev16 \u00b7 bundle 4",
      "macro_f1": 0.40554239377337675,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9706538766926329,
      "C": 0,
      "D": 0
    },
    {
      "method": "HGB-Dev16 \u00b7 bundle 4",
      "macro_f1": 0.4055379330496064,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9706581686302281,
      "C": 0,
      "D": 2
    },
    {
      "method": "Logistic-Global4 \u00b7 bundle 4",
      "macro_f1": 0.40554239377337675,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9706538766926329,
      "C": 0,
      "D": 0
    },
    {
      "method": "HGB-Global4 \u00b7 bundle 4",
      "macro_f1": 0.4055379330496064,
      "malicious_recall": 1.0,
      "false_positive_rate": 0.9706581686302281,
      "C": 0,
      "D": 2
    }
  ]
};
