#' DILI prediction model with Stan
#'
#' @export
#' @param x Numeric vector of input values.
#' @param y Numberic vector of output values.
#' @param ... Arguments passed to `rstan::sampling` (e.g. iter, chains).
#' @return An object of class `stanfit` returned by `rstan::sampling`
#'
#'

dili.prediction <- function(name = "ABC",
                            id = "HTL123",
                            BSEP = 314,
                            Glu = 250,
                            Glu.Gal = 1,
                            ClogP = 2.5,
                            BA = -1,        # BA should be either -1 or 1
                            cmax = 3       # Untransformed CMax values should be entered. Multiple values can be added
                            ) {

  ## read in data
  d2 <- structure(list(Drug = structure(1:96, .Label = c("Acetaminophen",
                                                         "Acetylsalicylic acid", "Acyclovir", "Albuterol", "Alendronate",
                                                         "Ambrisentan", "Aminopyrine", "Amiodarone", "Amlodipine", "Amodiaquine",
                                                         "Benoxaprofen", "Benzbromarone", "Benztropine", "Biperiden",
                                                         "Bosentan", "Bromfenac", "Bumetanide", "Buspirone", "Captopril",
                                                         "Carbamazepine", "Celecoxib", "Chlorpromazine", "Clomipramine",
                                                         "Clozapine", "Cycloserine", "Dantrolene", "Desipramine", "Dexamethasone",
                                                         "Diclofenac", "Entacapone", "Ethotoin", "Ezetimibe", "Felbamate",
                                                         "Felodipine", "Fenclozic Acid", "Flavoxate", "Flucloxacillin",
                                                         "Fludarabine", "Flumazenil", "Fluoxetine", "Flutamide", "Furazolidone",
                                                         "Guanethidine", "Ibufenac", "Ibuprofen", "Imipramine", "Indomethacin",
                                                         "Indoramin", "Itraconazole", "Ketoconazole", "Lapatinib", "Liothyronine",
                                                         "Mecamylamine", "Meclofenamate", "Metergoline", "Methotrexate",
                                                         "Nadolol", "Naproxen", "Nefazodone", "Neostigmine", "Nicardipine",
                                                         "Nifedipine", "Nimesulide", "Nitrofurantoin", "Olanzapine", "Orphenadrine",
                                                         "Oxybutynin", "Pargyline", "Paroxetine", "Perhexiline", "Phenoxybenzamine",
                                                         "Pimozide", "Pioglitazone", "Procyclidine", "Propantheline",
                                                         "Pyridostigmine", "Rimonabant", "Rosiglitazone", "Simvastatin",
                                                         "Sitaxsentan", "Stavudine", "Sudoxicam", "Sunitinib", "Suprofen",
                                                         "Tacrine", "Tamoxifen", "Ticlopidine", "Tienilic acid", "Tolcapone",
                                                         "Tolmetin", "Troglitazone", "Verapamil", "Warfarin", "Ximelagatran",
                                                         "Zileuton", "Zomepirac"), class = "factor"),
                       DILI.severity.category = c(3, 3, 3, 5, 5, 5, 3, 1, 3, 2, 1, 1, 5, 5, 1, 1, 4, 4, 2, 2, 2, 3,
                                                  3, 2, 4, 1, 3, 4, 2, 4, 4, 3, 1, 4, 1, 5, 2, 4, 5, 3, 1, 3, 5,
                                                  1, 3, 2, 2, 5, 2, 1, 1, 5, 5, 4, 5, 1, 4, 2, 1, 5, 4, 3, 2, 2,
                                                  3, 5, 5, 4, 2, 1, 5, 3, 3, 5, 5, 5, 5, 2, 2, 1, 1, 1, 1, 3, 2,
                                                  2, 2, 1, 1, 2, 1, 3, 3, 1, 2, 5),
                       log10.cmax = c(1.43232582762107, 2.0890033235067, 0.802065632544238, -1.18558384116108, -1.67754655284004,
                                      0.0528186583226165, 1.43004833349114, 0.243631196877119, -1.40666883289357,
                                      -1.00640137658079, 1.43301785015981, 0.76050739177983, -0.514149290134241,
                                      -0.721369806651315, 0.362309777405404, 0.589550951147358, -1.18058557341417,
                                      -1.90512322266609, 0.171943985761977, 1.02514113837999, -0.00734754431555313,
                                      -0.174370463183616, -0.409075355007302, -0.0207729128636299,
                                      1.67375317004584, 0.383338116152266, -0.112348675218606, -0.845188548664445,
                                      0.469095244047513, 0.0796970343467747, 1.70746234083202, -0.895692526360082,
                                      1.73586742612334, -1.52509326931349, 1.81158410710475, -0.129070702463331,
                                      1.11600582963822, -0.940600209627321, -0.290117158722675, -1.37331865556852,
                                      0.25406509449785, -0.187380713567005, -1.35983329399257, 1.44994003171079,
                                      1.38798724220826, -0.545303221015288, 0.403879980592733, -0.759094694627287,
                                      0.0297960548728227, 0.506495160887702, 0.686372354783051, -1.89281249074773,
                                      -1.00384345812359, 1.18644208548978, -1.24862523850973, 0.225237351873072,
                                      -0.403051823779777, 1.66608712732463, 0.169944650865884, -1.40075241807704,
                                      -1.19496969298864, -0.757971187273713, 0.721889918652702, 0.720440913292974,
                                      -1.27754034613442, -0.432253115815173, -1.64003825511036, -0.327858415454046,
                                      -0.884624659016221, -0.0635573743252597, -0.88900003562646, -1.09081972580418,
                                      0.0430992283977766, -0.0588801891772476, -0.612972858793975,
                                      -0.295498317078943, -0.627090630497662, -0.314817481156277, -1.61747931851363,
                                      0.826800544283902, 0.0988660686925641, 1.16894828527586, 1.13970650060147,
                                      1.29486105283451, -1.20197966570263, -0.227079616373813, 0.390883288367234,
                                      1.19344477396761, 1.00260508220023, 1.65007769571644, 0.309998466015901,
                                      -0.535764563547608, 0.74742751974549, -0.604933355397073, 0.558284317825288,
                                      0.573502547612699),
                       BSEP = c(1.33690140582683, 1.33690140582683, -0.757647531772657, 1.33690140582683, 1.33690140582683, 0.744754264235985,
                                1.33690140582683, -0.719840658560066, -0.68179297723784, -0.381986880743088,
                                -0.519969927563564, -1.03722574705812, 0.905132465252455, 0.528749389893983,
                                -0.96811381959307, -0.644467720244518, 0.902965192265746, -0.733325912699589,
                                1.33690140582683, 1.33690140582683, -0.97148513312795, -0.570298822477141,
                                -0.473012346184868, -0.731881064041783, 1.33690140582683, -0.72393439642385,
                                1.33690140582683, -0.305169093769734, -0.710208334174692, -0.652414387862451,
                                1.33690140582683, -1.00640230902493, 1.33690140582683, -1.02976069565946,
                                1.33690140582683, -0.89322249749679, -1.06973484185876, 1.33690140582683,
                                1.33690140582683, 0.0112527622897879, -0.750664096593261, -0.830371580882228,
                                1.33690140582683, 1.33690140582683, -0.433279008095202, 0.784246794216017,
                                -0.631945698543532, 0.240983698880948, -1.062992214789, -1.06130655802156,
                                -1.06130655802156, -0.874439464945312, -0.768002280486934, -0.885275829878857,
                                -0.969077052031607, 1.33690140582683, 1.33690140582683, -0.167186046949257,
                                -1.06082494180229, 1.33690140582683, -1.05576797149997, -0.945477857287442,
                                -0.497815581477205, 1.33690140582683, -0.793287131998539, 1.33690140582683,
                                -0.863603100011766, 1.33690140582683, -0.10240866545762, -0.697204696254438,
                                -0.630260041776092, -0.909356640842291, -1.07045726618766, 1.33690140582683,
                                1.33690140582683, 1.33690140582683, -1.04589483900496, -1.05576797149997,
                                -1.03120554431727, -0.92260108687218, 1.33690140582683, -0.25170969343091,
                                -0.928621289613038, 1.33690140582683, 1.33690140582683, -0.767520664267665,
                                -0.863121483792498, 0.442058470425621, -0.82218410515466, 1.33690140582683,
                                -1.06491867966607, -0.788470969805853, -0.676495198825885, -0.449172343331069,
                                1.33690140582683, -0.295295961274726),
                       Glu = c(-0.0194936215750834,
                                0.620126196621512, 0.620126196621512, 0.620126196621512, 0.620126196621512,
                                0.620126196621512, 0.620126196621512, -1.08136246037802, -2.25066744051868,
                                0.620126196621512, 0.620126196621512, -0.0669654049568621, -1.13133275867463,
                                0.620126196621512, 0.620126196621512, 0.620126196621512, 0.620126196621512,
                                0.620126196621512, 0.620126196621512, 0.592642532558377, -2.04079218767292,
                                -2.04578921750258, -2.16696719087186, -0.771546610939049, 0.620126196621512,
                                0.620126196621512, -1.89088129278309, 0.620126196621512, 0.620126196621512,
                                0.620126196621512, 0.620126196621512, 0.620126196621512, 0.620126196621512,
                                -0.118184960710886, 0.620126196621512, 0.620126196621512, 0.620126196621512,
                                0.620126196621512, 0.620126196621512, -2.21693748916846, 0.508942282911557,
                                -0.106941643594149, 0.620126196621512, 0.620126196621512, -0.676603044175492,
                                -1.35245132863713, 0.620126196621512, 0.620126196621512, 0.620126196621512,
                                0.302814802438044, -2.14448055663838, 0.620126196621512, 0.620126196621512,
                                0.620126196621512, -0.83151096889498, -2.260661500178, 0.620126196621512,
                                0.620126196621512, -0.847751315841378, 0.620126196621512, 0.580149957984225,
                                0.620126196621512, 0.620126196621512, 0.551417036463674, 0.620126196621512,
                                0.620126196621512, -2.19070308256274, 0.620126196621512, -2.31437957084685,
                                -2.35560506694155, -0.124431247997962, -1.97208302751508, 0.620126196621512,
                                0.496449708337404, 0.620126196621512, 0.620126196621512, -0.532938436572741,
                                0.620126196621512, -1.03264141953883, 0.620126196621512, 0.620126196621512,
                                0.620126196621512, -1.96958451260025, 0.620126196621512, 0.426491290722152,
                                -1.64227905875746, 0.473963074103931, 0.620126196621512, -0.22187332967635,
                                0.620126196621512, -0.380529026768084, -0.521695119456004, 0.620126196621512,
                                0.620126196621512, 0.620126196621512, 0.620126196621512),
                       Glu.Gal = c(-0.432432514277011,
                                  -0.432432514277011, -0.432432514277011, -0.432432514277011, -0.432432514277011,
                                  -0.432432514277011, -0.432432514277011, 0.495984896566924, -0.565063572969002,
                                  -0.432432514277011, -0.0345393382010391, 0.628615955258915, 1.29177124871887,
                                  -0.432432514277011, -0.432432514277011, -0.432432514277011, -0.432432514277011,
                                  -0.0345393382010391, -0.432432514277011, -0.432432514277011,
                                  1.95492654217882, -0.299801455585021, -0.565063572969002, -0.432432514277011,
                                  -0.432432514277011, -0.432432514277011, -0.299801455585021, -0.432432514277011,
                                  -0.432432514277011, 0.761247013950905, -0.432432514277011, 0.363353837874933,
                                  -0.432432514277011, 0.893878072642896, -0.432432514277011, -0.432432514277011,
                                  -0.432432514277011, 0.363353837874933, -0.432432514277011, -0.432432514277011,
                                  0.761247013950905, 4.87280983340262, -0.432432514277011, -0.432432514277011,
                                  -0.962956749044974, -0.299801455585021, -0.432432514277011, -0.432432514277011,
                                  2.08755760087081, 0.230722779182942, 2.35281971825479, -0.432432514277011,
                                  -0.432432514277011, 0.363353837874933, 0.495984896566924, -1.36084992512095,
                                  -0.432432514277011, -0.432432514277011, 5.1380719507866, -0.432432514277011,
                                  1.42440230741086, 0.761247013950905, -0.16717039689303, 1.55703336610285,
                                  -0.432432514277011, -0.432432514277011, -0.962956749044974, -0.432432514277011,
                                  -0.830325690352984, 0.0980917204909515, -0.0345393382010391,
                                  1.15914019002688, -0.432432514277011, -0.432432514277011, -0.432432514277011,
                                  -0.432432514277011, 1.82229548348683, -0.432432514277011, -0.16717039689303,
                                  -0.432432514277011, -0.16717039689303, -0.432432514277011, -0.565063572969002,
                                  -0.432432514277011, -0.0345393382010391, -0.16717039689303, -0.16717039689303,
                                  -0.432432514277011, 0.893878072642896, -0.432432514277011, -0.299801455585021,
                                  -0.299801455585021, -0.432432514277011, -0.432432514277011, -0.432432514277011,
                                  -0.432432514277011),
                       ClogP = c(-1.02695635990943, -0.823346672690704,
                                 -2.20789254577801, -1.1898441096844, -3.51099454397782, 0.316867575734135,
                                 0.276145638290391, 2.39368638536509, 0.153979825959158, 1.00914051227779,
                                 0.316867575734135, 1.21275019949651, 0.194701763402902, 0.764808887615322,
                                 0.479755325509112, 0.113257888515414, 0.153979825959158, -0.334683423365773,
                                 -0.864068610134448, -0.253239548478285, 0.561199200396601, 0.927696637390299,
                                 1.17202826205276, 0.276145638290391, -1.71922929645308, -0.579015048028239,
                                 0.601921137840345, -0.49757117314075, 0.683365012727833, -0.49757117314075,
                                 -0.619736985471983, 0.398311450621623, -1.02695635990943, 0.927696637390299,
                                 -0.171795673590796, 0.805530825059066, -0.131073736147052, -2.04500479600303,
                                 -0.701180860359471, 0.642643075284089, 0.113257888515414, -1.23056604712815,
                                 -0.660458922915727, 0.153979825959158, 0.276145638290391, 0.805530825059066,
                                 0.479755325509112, -0.0903517987033077, 1.21275019949651, 0.235423700846646,
                                 1.21275019949651, -0.171795673590796, -0.0903517987033077, 1.37563794927149,
                                 0.683365012727833, -1.43417573434687, -1.06767829735317, -1.39345379690312,
                                 1.09058438716528, -2.37078029555299, 0.886974699946554, 0.0318140136279252,
                                 0.0725359510716695, -1.43417573434687, -0.00890792381581909,
                                 0.357589513177879, 0.764808887615322, -0.171795673590796, 0.479755325509112,
                                 1.70141344882144, 0.764808887615322, 1.37563794927149, 0.194701763402902,
                                 0.642643075284089, -0.619736985471983, -2.98160935720915, 1.41635988671523,
                                 -0.00890792381581909, 0.601921137840345, 0.153979825959158, -1.43417573434687,
                                 -0.741902797803216, -0.00890792381581909, -0.21251761103454,
                                 0.113257888515414, 1.53852569904646, 0.561199200396601, 0.0725359510716695,
                                 0.0725359510716695, -0.334683423365773, 1.04986244972153, 0.601921137840345,
                                 -0.0496298612595634, -0.49757117314075, -0.21251761103454, -0.0496298612595634
                                  ),
                       dili.sev = c(2, 2, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 1, 1, 3,
                                    3, 1, 1, 2, 2, 2, 2, 2, 2, 1, 3, 2, 1, 2, 1, 1, 2, 3, 1, 3, 1,
                                    2, 1, 1, 2, 3, 2, 1, 3, 2, 2, 2, 1, 2, 3, 3, 1, 1, 1, 1, 3, 1,
                                    2, 3, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 3, 1, 2, 2, 1, 1, 1, 1, 2,
                                    2, 3, 3, 3, 3, 2, 2, 2, 2, 3, 3, 2, 3, 2, 2, 3, 2, 1),
                       BA = c(1,
                              -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1,
                              1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1,
                              -1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, -1, 1,
                              -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1,
                              -1, -1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1,
                              1, 1),
                       cols = structure(c(1L, 1L, 1L, 3L, 3L, 3L, 1L, 2L, 1L,
                                          1L, 2L, 2L, 3L, 3L, 2L, 2L, 3L, 3L, 1L, 1L, 1L, 1L, 1L, 1L, 3L,
                                          2L, 1L, 3L, 1L, 3L, 3L, 1L, 2L, 3L, 2L, 3L, 1L, 3L, 3L, 1L, 2L,
                                          1L, 3L, 2L, 1L, 1L, 1L, 3L, 1L, 2L, 2L, 3L, 3L, 3L, 3L, 2L, 3L,
                                          1L, 2L, 3L, 3L, 1L, 1L, 1L, 1L, 3L, 3L, 3L, 1L, 2L, 3L, 1L, 1L,
                                          3L, 3L, 3L, 3L, 1L, 1L, 2L, 2L, 2L, 2L, 1L, 1L, 1L, 1L, 2L, 2L,
                                          1L, 2L, 1L, 1L, 2L, 1L, 3L),
                                        .Label = c("darkgoldenrod", "firebrick",
                                                  "green4"), class = "factor")), row.names = c(NA, -96L),
                              spec = structure(list(cols = list(Drug = structure(list(),
                                                                                 lass = c("collector_character",
                                                                                                   "collector")),
                                                                DILI.severity.category = structure(list(), class = c("collector_double", "collector")),
                                                                log10.cmax = structure(list(), class = c("collector_double", "collector")),
                                                                Spher = structure(list(), class = c("collector_double",  "collector")),
                                                                BSEP = structure(list(), class = c("collector_double", "collector")),
                                                                THP1 = structure(list(), class = c("collector_double", "collector")),
                                                                Glu = structure(list(), class = c("collector_double",  "collector")),
                                                                Gal = structure(list(), class = c("collector_double", "collector")),
                                                                Glu.Gal = structure(list(), class = c("collector_double",  "collector")),
                                                                ClogP = structure(list(), class = c("collector_double", "collector")),
                                                                dili.sev = structure(list(), class = c("collector_double", "collector")),
                                                                BA = structure(list(), class = c("collector_double", "collector")),
                                                                cols = structure(list(), class = c("collector_character", "collector"))),
                                                    default = structure(list(), class = c("collector_guess", "collector")), skip = 1), class = "col_spec"),
                  class = "data.frame")

  ## ---------------------------------------------------------
  ## Define and fit model
  ## ---------------------------------------------------------

  ## specify design matrix (intercept not needed; included as cutpoints)
  ## Spher and THP1 removed
  X <- model.matrix(~ 0 + (BSEP + Glu + Glu.Gal + ClogP + BA)^2 +
                      log10.cmax, data=d2)

  ## draw samples from posterior
  m1 <- rstan::sampling(stanmodels$MLmodel,
                 data=list(N=nrow(X),
                           P=ncol(X),
                           y=d2$dili.sev,
                           X=X,
                           sigma_prior=1,
                           make_pred = 1,
                           N_pred=nrow(X), # (predict the training data)
                           X_pred=X),
                 iter=10000, chains=3, seed=123)

  ## extract posterior samples
  pr <- rstan::extract(m1)

  ## extract predicted (training) values and convert to 0-1 scale
  post <- data.frame(invlogit(pr$eta))

  ## mean cutpoints
  c1 <- mean(invlogit(pr$cutpoints[,1]))
  c2 <- mean(invlogit(pr$cutpoints[,2]))

  ## calculate average profile for each of the 3 DILI categories
  avg <- post %>%
    purrr::map( ~ density(., adjust=1.5, from=0, to=1, n=512)$y) %>%
    as.data.frame() %>%
    tidyr::gather(value = "y") %>%
    dplyr::mutate(sev = rep(d2$dili.sev, each = 512),
           x = rep(seq(0, 1, length.out = 512), nrow(d2))) %>%
    dplyr::group_by(sev, x) %>%
    dplyr::summarise(m = mean(y))

  ## data for two new compounds

  new.compounds <- data.frame(name = name,
                              id = id,
                              BSEP = BSEP,
                              Glu = Glu,
                              Glu.Gal = Glu.Gal,
                              ClogP = ClogP,
                              BA = BA,
                              log10.cmax = log10(cmax) )

  ## standardise
  new.cpds.scaled <- recipes::bake(parms, new_data = new.compounds, composition="data.frame") %>%
    dplyr::mutate(Drug = name)

  ## ---------------------------------------------------------
  ## Prediction on new compounds
  ## ---------------------------------------------------------

  ## generate design matrix
  X.new <- model.matrix(~ 0 + (BSEP + Glu + Glu.Gal + ClogP + BA)^2 +
                          log10.cmax, data=new.cpds.scaled)


  ## fit model to training data (again) and predict  test data
  m2 <- rstan::sampling(stanmodels$MLmodel,
                 data=list(N=nrow(X),
                           P=ncol(X),
                           y=d2$dili.sev,
                           X=X,
                           sigma_prior=1,
                           make_pred = 1,
                           N_pred=nrow(X.new), # (predict test data)
                           X_pred=X.new),
                 iter=10000, chains=3, seed=123)

  ## extract posterior samples
  pr.test <- rstan::extract(m2)

  return(pr.test)

}
