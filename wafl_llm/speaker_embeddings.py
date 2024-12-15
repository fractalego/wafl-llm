import torch


speaker_embedding = torch.tensor(
    [[-0.08344213664531708, -0.020733917132019997, 0.04722348600625992, 0.03869166597723961, 0.040116727352142334,
      -0.026968270540237427, -0.05174599587917328, -0.05634180083870888, 0.035828497260808945, 0.017309317365288734,
      -0.03477650508284569, -0.04979589954018593, 0.062050383538007736, 0.03760506212711334, 0.014607330784201622,
      0.0408836230635643, -0.01127066183835268, 0.0072131529450416565, 0.015619106590747833, 0.019706321880221367,
      0.03903719037771225, 0.010859797708690166, -0.015195660293102264, -0.05317859724164009, -0.06590470671653748,
      -0.009104733355343342, -0.0628313273191452, 0.020135484635829926, 0.05084865167737007, 0.025366179645061493,
      -0.001995180267840624, 0.03163020312786102, 0.05029085651040077, -0.030064569786190987, 0.005786243826150894,
      -0.08004838228225708, 0.03636869788169861, 0.0726659819483757, -0.03291487693786621, -0.06725309044122696,
      0.0023526570294052362, -0.0314236544072628, 0.054618123918771744, 0.021657316014170647, 0.037761908024549484,
      -0.1238769069314003, -0.018349625170230865, 0.01196556631475687, -0.07752393186092377, 0.0623108372092247,
      0.0031452204566448927, 0.03057968243956566, 0.012076998129487038, 0.026150180026888847, -0.07762616872787476,
      0.006668524816632271, 0.028903353959321976, 0.036332786083221436, 0.0038808071985840797, 0.02220206707715988,
      0.023975273594260216, -0.009341815486550331, -0.014527909457683563, 0.05196438729763031, -0.011377363465726376,
      0.02871144935488701, 0.02881045453250408, -0.029625773429870605, -0.06660275906324387, -0.0708247646689415,
      0.02808437868952751, 0.010311804711818695, -0.035429224371910095, 0.019216977059841156, 0.03565952926874161,
      0.02358727529644966, 0.00804113782942295, 0.03211146965622902, -0.043329376727342606, -0.0883973017334938,
      -0.04288102313876152, -0.05429692938923836, -0.07715139538049698, -0.061414629220962524, -0.03188523277640343,
      -0.0652315691113472, -0.07850759476423264, 0.046123553067445755, 0.017187315970659256, 0.05029035732150078,
      0.027183257043361664, -0.08996610343456268, 0.016073204576969147, -0.05004711449146271, 0.029959574341773987,
      0.0181458480656147, 0.007732781581580639, 0.045994438230991364, -0.048298489302396774, -0.10461872071027756,
      0.030885886400938034, -0.04736986383795738, -0.07551990449428558, 0.03291676193475723, 0.03665011748671532,
      -0.04541151225566864, 0.03167841210961342, 0.041052669286727905, 0.022580919787287712, 0.009800461120903492,
      -0.07953444868326187, 0.05836549773812294, 0.06462753564119339, 0.0070035685785114765, 0.04220540449023247,
      0.06059672683477402, -0.060787100344896317, -0.03460048884153366, -0.05484018474817276, -0.03279876336455345,
      0.004894405137747526, -0.04750774800777435, 0.019871585071086884, 0.02004030905663967, -0.06005992740392685,
      0.056557320058345795, -0.10351908206939697, 0.01754124090075493, 0.04153290390968323, 0.030254339799284935,
      0.005929615348577499, 0.030397770926356316, 0.030766932293772697, 0.0837191492319107, 0.011049334891140461,
      -0.07465492188930511, -0.05398757755756378, 0.03191591426730156, -0.08306196331977844, -0.06013287231326103,
      0.026253674179315567, 0.0031975461170077324, -0.026167718693614006, 0.024982187896966934, -0.05898144841194153,
      0.03810352832078934, 0.025317206978797913, -0.016650089994072914, 0.06656645983457565, -0.05984332039952278,
      0.06993285566568375, -0.03989287465810776, -0.05597176030278206, 0.03585240617394447, 0.03340771421790123,
      0.011679606512188911, 0.012570912949740887, -0.060307297855615616, -0.06563588976860046, 0.04421526566147804,
      0.03571692109107971, -0.01435155887156725, 0.015733392909169197, -0.06336212903261185, -0.0015778777888044715,
      0.011190947145223618, 0.04939202219247818, 0.022474372759461403, 0.016257403418421745, 0.041050348430871964,
      0.010630296543240547, -0.07075928151607513, 0.027853835374116898, -0.07144850492477417, -0.04206368327140808,
      -0.012382605113089085, -0.0328034833073616, 0.02740505337715149, 0.04718601703643799, -0.06425002962350845,
      0.026298241689801216, -0.055767741054296494, 0.011777556501328945, 0.049009062349796295, -0.04694148898124695,
      -0.00858271773904562, 0.030544806271791458, 0.026679011061787605, -0.03974888101220131, -0.05817654728889465,
      0.019479775801301003, 0.018648246303200722, -0.07586979866027832, 0.03648035600781441, 0.01971563510596752,
      0.010289494879543781, 0.029910199344158173, 0.008311210200190544, 0.06674645096063614, -0.010108459740877151,
      0.010074781253933907, 0.00956330168992281, -0.043471239507198334, 0.005105751100927591, 0.02269880659878254,
      0.01640278659760952, 0.02861163765192032, -0.03245391696691513, 0.032636821269989014, -0.06082547456026077,
      0.0036232792772352695, -0.04731956869363785, -0.060963649302721024, 0.048825088888406754, -0.07243110239505768,
      0.03446178138256073, 0.043514762073755264, -0.06452246755361557, -0.0021488755010068417, 0.04055284708738327,
      -0.07183869928121567, 0.026236480101943016, -0.07007148861885071, -0.03951999545097351, 0.035149917006492615,
      0.05264906957745552, 0.07125718146562576, 0.09420765191316605, 0.03623354434967041, 0.020783064886927605,
      -0.03063533641397953, 0.034766923636198044, 0.017699098214507103, -0.026238353922963142, 0.00735819386318326,
      0.005341134034097195, 0.01885341666638851, 0.011666730046272278, -0.0013159291120246053, 0.06898827105760574,
      -0.04822279140353203, 0.02650323323905468, -0.07366843521595001, 0.013853518292307854, -0.07107984274625778,
      0.03279130160808563, 0.01808803342282772, 0.028085995465517044, 0.02211497537791729, 0.046455640345811844,
      0.030590973794460297, 0.020745545625686646, 0.00984905008226633, -0.060658618807792664, 0.0018757088109850883,
      0.03415822237730026, 0.006284649949520826, 0.02268781140446663, -0.04434967041015625, 0.05027838796377182,
      0.0219328124076128, -0.08249891549348831, 0.014917192980647087, 0.03790958598256111, 0.0006323581910692155,
      0.02244776301085949, 0.0438808836042881, 0.008692081086337566, -0.004916156176477671, -0.057509325444698334,
      0.04766986519098282, 0.005911092273890972, 0.022291913628578186, -0.039100281894207, -0.06785783916711807,
      0.026442501693964005, 0.012451783753931522, 0.024378227069973946, -0.11976409703493118, -0.04942450299859047,
      -0.006654651835560799, 0.03375981003046036, 0.025005921721458435, -0.06852120906114578, -0.04455356299877167,
      0.04522326588630676, 0.014392146840691566, 0.056229040026664734, -0.09053078293800354, 0.0037506951484829187,
      0.012740186415612698, 0.007498341612517834, -0.05041762813925743, 0.029726577922701836, -0.047701746225357056,
      0.02423124760389328, 0.011050550267100334, 0.02834879793226719, 0.0432131290435791, 0.011355441063642502,
      0.044916845858097076, 0.011713674291968346, 0.018617946654558182, -0.052869684994220734, 0.009766584262251854,
      0.051095105707645416, 0.02807939052581787, 0.01638488844037056, 0.03193815052509308, 0.026865139603614807,
      0.011800440028309822, 0.03306206688284874, -0.055764541029930115, 0.06705047190189362, -0.08239731192588806,
      -0.06351208686828613, -0.026306068524718285, 0.020929111167788506, -0.06077833101153374, 0.052341945469379425,
      -0.04563678801059723, 0.03605979308485985, -0.0387267991900444, 0.005794320721179247, 0.047175388783216476,
      -0.05200214684009552, 0.05187224969267845, 0.08167266845703125, -0.10130263864994049, -0.08458177000284195,
      -0.040086932480335236, 0.04226839542388916, -0.0016622365219518542, 0.026222052052617073, -0.0808987021446228,
      0.055342305451631546, 0.052592892199754715, -0.04124418646097183, -0.006752749439328909, -0.05075911805033684,
      0.024616459384560585, -0.06835555285215378, 0.017546480521559715, -0.006795234512537718, -0.07186141610145569,
      0.010383937507867813, 0.008118776604533195, 0.041702751070261, -0.040661245584487915, -0.07335308194160461,
      -0.037315499037504196, -0.04197518527507782, -0.03634686768054962, 0.10523095726966858, 0.03646691143512726,
      -0.04302534833550453, -0.007875747978687286, 0.034993309527635574, -0.04831598326563835, 0.015499615110456944,
      -0.09440234303474426, 0.02458804100751877, -0.005612069275230169, 0.014273406006395817, 0.051838744431734085,
      -0.019709324464201927, -0.05788980796933174, 0.015095815062522888, 0.0030642955098301172, 0.048376042395830154,
      0.012413578107953072, 0.05800040066242218, -0.07378330081701279, -0.020665181800723076, -0.011487126350402832,
      -0.0025882006157189608, 0.057451069355010986, 0.009903084486722946, 0.04442485049366951, 0.042657315731048584,
      -0.06865114718675613, -0.10424432158470154, 0.016737449914216995, 0.052158698439598083, 0.0484788715839386,
      -0.08156794309616089, 0.024786846712231636, -0.055382125079631805, 0.04100732132792473, 0.03801076486706734,
      0.02748514898121357, -0.0013883989304304123, -0.044979602098464966, 0.03313426673412323, 0.02366732433438301,
      0.004511956125497818, 0.030766664072871208, -0.10031505674123764, -0.012440681457519531, -0.03644053265452385,
      -0.01003227848559618, -0.053148187696933746, -0.01701486110687256, -0.02330239675939083, 0.020686158910393715,
      0.0025947110261768103, 0.014991945587098598, -0.05561164766550064, -0.05838322639465332, -0.06251442432403564,
      -0.07294856011867523, 0.02481461688876152, 0.015727171674370766, -0.014725365675985813, 0.011468274518847466,
      -0.050164807587862015, -0.010367187671363354, -0.0875377282500267, -0.13132591545581818, -0.03325987234711647,
      -0.042107854038476944, 0.01294784527271986, 0.04198865965008736, -0.04608660563826561, -0.054874613881111145,
      0.014981843531131744, 0.07887257635593414, 0.038094885647296906, 0.027336036786437035, -0.028457975015044212,
      -0.041706934571266174, -0.02047859877347946, -0.02621961198747158, 0.034052833914756775, 0.05442943051457405,
      0.0033619327004998922, 0.03908940777182579, -0.003512938506901264, -0.06401319056749344, -0.05314352735877037,
      0.010144779458642006, 0.008746272884309292, 0.036878399550914764, 0.0054969205521047115, 0.04168131947517395,
      0.03211384266614914, 0.0339711494743824, 0.011362781748175621, 0.0025082372594624758, -0.016057534143328667,
      0.050691209733486176, 0.0375710092484951, 0.034671857953071594, 0.05433927848935127, 0.004589141346514225,
      0.0028407075442373753, 0.003601068165153265, -0.07642342895269394, -0.051374584436416626, 0.036027297377586365,
      -0.01578723080456257, 0.010754824616014957, 0.03227930888533592, 0.03201024606823921, -0.0706285908818245,
      0.03299366310238838, 0.021211229264736176, -0.04480285197496414, 0.036275677382946014, 0.028675829991698265,
      -0.0025707571767270565, 0.038217321038246155, 0.019114937633275986, 0.004243936389684677, -0.02450643852353096,
      -0.06325725466012955, -0.003100512083619833, -0.004279746674001217, 0.0427163727581501, 0.035983551293611526,
      -0.06459880620241165, -0.04173222929239273, 0.008639545179903507, 0.014417647384107113, 0.03649972006678581,
      0.04982951283454895, -0.06292441487312317, 0.023404039442539215, -0.0020426602568477392, -0.012309598736464977,
      -0.041042063385248184, 0.023974992334842682, -0.05578097701072693, 0.017135612666606903, 0.008463526144623756,
      0.013907218351960182, -0.0711086243391037, 0.04800118878483772, -0.08502800762653351, -0.05524006858468056,
      0.009336283430457115, 0.038350559771060944, -0.0755411759018898, 0.0185368824750185, -0.05329255014657974,
      -0.012607692740857601, -0.0686722844839096, 0.04657493159174919, 0.07921521365642548, 0.019740767776966095,
      0.05248761177062988, -0.05411471426486969]],
  dtype=torch.bfloat16,
)