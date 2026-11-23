C
C  run_penelope.f
C
C  PENELOPE driver: generates positron transport raw data for the
C  condensed-history (HMC) engine's lookup tables.
C
C  For each material, for each energy, simulates NHIST positrons through
C  a step of length STEP_CM, recording per-history scattering angle (theta),
C  kinetic-energy loss (dE_kin), total deposited energy, annihilation
C  flag, and the end position relative to the start (z along the initial
C  direction, r perpendicular to it) -- for annihilated histories that is
C  where the positron stopped inside the step -- and dphi, the azimuth of
C  the final direction about the initial one measured from the lateral
C  displacement (10th column, since 2026-10-30).
C
C  Configuration is read from 'input/sim.in' (step distance, energy
C  grid, histories, material folder). An optional keyword line
C  "SEED n" after the six positional lines sets the seed base: material
C  N is simulated with ISEED1 = ISEED2 = n + N (default n = 4000), the
C  same rule as the GEANT4 driver. Other keyword lines (GEANT4-only
C  options) are ignored. Materials are discovered
C  automatically from MAT_DIR/mat1.mat, mat2.mat, ... until a file is
C  not found.
C
C  Optional command-line arguments restrict the run to a range of
C  materials, so that several processes can share the work:
C    run_penelope.x            all materials
C    run_penelope.x 7          material 7 only
C    run_penelope.x 7 12       materials 7 .. 12
C  (run_step_variant.sh --jobs N starts N such processes.)
C
C  Output (per material):
C    output/penelope/raw/matXX.dat    -- per-history raw text
C    output/penelope/logs/run_penelope_matXX.err -- PENELOPE warnings
C                                              (deleted when empty)
C
C  Compact binary tables are built from this raw text by
C  build_tables.py --engine penelope, the same compaction path used for
C  the GEANT4 engine's output, so both engines share a single (tested)
C  implementation of the event-subsampling logic.
C
C  PENELOPE source is included from the PenSuite-2023 distribution; the
C  -I flag in COMPILAR.sh points the compiler at the PenSuite fsource/
C  directory.
C
      INCLUDE 'penelope.f'
      INCLUDE 'rita.f'

      PROGRAM POSITRON_TABLEGEN
      USE TRACK_mod
      USE PENELOPE_mod
      USE PENERROR_mod
      IMPLICIT DOUBLE PRECISION (A-H,O-Z), INTEGER*4 (I-N)
      LOGICAL LANN, LEXIST
      INTEGER IOS
      COMMON/RSEED/ISEED1,ISEED2

C  ****  Compile-time limits.
      INTEGER MAXEN, MAXMATL
      PARAMETER (MAXEN=500)
      PARAMETER (MAXMATL=200)

C  ****  Config variables (read from input/sim.in).
      DOUBLE PRECISION STEP_CM, EMIN_SIM, EMAX_SIM, EMAX_GRID
      INTEGER NENRG, NHIST, NMATS, ISEEDB
      INTEGER MFIRST, MLAST, NARG, ISZ
      CHARACTER*80 MATDIR
      CHARACTER*32 CARG
      CHARACTER*120 FNAME, RFILE, EFILE
      CHARACTER*120 KLINE
      CHARACTER*20 PMFILE(MAXMATL)

C  ****  Energy grid.
      DIMENSION ENERGY(MAXEN)

C  ****  Read configuration. -----------------------------------------------
      OPEN(10,FILE='input/sim.in',STATUS='OLD',IOSTAT=IOS)
      IF(IOS.NE.0)THEN
        WRITE(6,*) 'ERROR: cannot open input/sim.in'
        STOP 1
      ENDIF
      READ(10,*,IOSTAT=IOS) STEP_CM
      IF(IOS.NE.0) GOTO 7100
      READ(10,*,IOSTAT=IOS) EMIN_SIM
      IF(IOS.NE.0) GOTO 7100
      READ(10,*,IOSTAT=IOS) EMAX_SIM
      IF(IOS.NE.0) GOTO 7100
      READ(10,*,IOSTAT=IOS) NENRG
      IF(IOS.NE.0) GOTO 7100
      IF(NENRG.LT.1 .OR. NENRG.GT.MAXEN) THEN
        WRITE(6,*) 'ERROR: NENRG out of range (1..',MAXEN,')'
        STOP 1
      ENDIF
      READ(10,*,IOSTAT=IOS) NHIST
      IF(IOS.NE.0) GOTO 7100
      READ(10,'(A)',IOSTAT=IOS) MATDIR
      IF(IOS.NE.0) GOTO 7100
C  ****  Strip trailing comment (everything from '!' onwards).
      IC = INDEX(MATDIR,'!')
      IF(IC.GT.0) MATDIR(IC:80) = ' '
      MATDIR = ADJUSTL(TRIM(MATDIR))
C  ****  Optional keyword lines; only SEED is used by this driver.
      ISEEDB = 4000
      DO
        READ(10,'(A)',IOSTAT=IOS) KLINE
        IF(IOS.NE.0) EXIT
        IC = INDEX(KLINE,'!')
        IF(IC.GT.0) KLINE(IC:120) = ' '
        KLINE = ADJUSTL(KLINE)
        IF(KLINE(1:5).EQ.'SEED ' .OR. KLINE(1:5).EQ.'seed ')THEN
          READ(KLINE(6:120),*,IOSTAT=IOS) ISEEDB
          IF(IOS.NE.0 .OR. ISEEDB.LT.0) THEN
            WRITE(6,*) 'ERROR: bad SEED value in input/sim.in'
            STOP 1
          ENDIF
        ENDIF
      ENDDO
      CLOSE(10)
      GOTO 7101
 7100 WRITE(6,*) 'ERROR: cannot read input/sim.in'
      STOP 1
 7101 CONTINUE

      EMAX_GRID = DMAX1(EMAX_SIM*1.3D0, EMAX_SIM+1.0D6)

C  ****  Discover materials in MAT_DIR.
      NMATS = 0
      DO I = 1, MAXMATL
        IF(I.LT.10)THEN
          WRITE(FNAME,'(A,A,I1,A)') TRIM(MATDIR),'/mat',I,'.mat'
        ELSE IF(I.LT.100)THEN
          WRITE(FNAME,'(A,A,I2,A)') TRIM(MATDIR),'/mat',I,'.mat'
        ELSE
          WRITE(FNAME,'(A,A,I3,A)') TRIM(MATDIR),'/mat',I,'.mat'
        ENDIF
        INQUIRE(FILE=FNAME, EXIST=LEXIST)
        IF(.NOT.LEXIST) EXIT
        NMATS = NMATS + 1
      ENDDO
      IF(NMATS.EQ.0) THEN
        WRITE(6,*) 'ERROR: no .mat files found in ',TRIM(MATDIR)
        STOP 1
      ENDIF

C  ****  Optional material range on the command line.
      MFIRST = 1
      MLAST = NMATS
      NARG = COMMAND_ARGUMENT_COUNT()
      IF(NARG.GE.1) THEN
        CALL GET_COMMAND_ARGUMENT(1,CARG)
        READ(CARG,*,IOSTAT=IOS) MFIRST
        IF(IOS.NE.0) GOTO 7200
        MLAST = MFIRST
      ENDIF
      IF(NARG.GE.2) THEN
        CALL GET_COMMAND_ARGUMENT(2,CARG)
        READ(CARG,*,IOSTAT=IOS) MLAST
        IF(IOS.NE.0) GOTO 7200
      ENDIF
      IF(NARG.GT.2) GOTO 7200
      IF(MFIRST.LT.1 .OR. MLAST.GT.NMATS .OR. MFIRST.GT.MLAST) THEN
        WRITE(6,*) 'ERROR: material range ',MFIRST,'..',MLAST,
     1    ' outside 1..',NMATS
        STOP 1
      ENDIF
      GOTO 7201
 7200 WRITE(6,*) 'usage: run_penelope.x [MFIRST [MLAST]]'
      STOP 1
 7201 CONTINUE

      WRITE(6,*)'run_penelope configuration:'
      WRITE(6,*)'  STEP_CM   = ',STEP_CM,' cm'
      WRITE(6,*)'  EMIN_SIM  = ',EMIN_SIM,' eV'
      WRITE(6,*)'  EMAX_SIM  = ',EMAX_SIM,' eV'
      WRITE(6,*)'  NENRG     = ',NENRG
      WRITE(6,*)'  NHIST     = ',NHIST
      WRITE(6,*)'  MAT_DIR   = ',TRIM(MATDIR)
      WRITE(6,*)'  NMATS     = ',NMATS
      WRITE(6,*)'  MATERIALS = ',MFIRST,' ..',MLAST
      WRITE(6,*)'  SEED      = ',ISEEDB,' (+ material number)'
      WRITE(6,*)'--------------------------------------------'

C  ****  Generate logarithmic energy grid.
      FAC = LOG(EMAX_SIM/EMIN_SIM) / DBLE(NENRG-1)
      DO IE = 1, NENRG
        ENERGY(IE) = EMIN_SIM * EXP(DBLE(IE-1)*FAC)
      ENDDO

C  ****  Main material loop. ------------------------------------------------
      DO IMAT = MFIRST, MLAST
        WRITE(6,*)' Processing material',IMAT

        IF(IMAT.LT.10)THEN
          WRITE(FNAME,'(A,A,I1,A)') TRIM(MATDIR),'/mat',IMAT,'.mat'
          WRITE(RFILE,'(A,I1,A)')
     1      'output/penelope/raw/mat',IMAT,'.dat'
          WRITE(EFILE,'(A,I1,A)')
     1      'output/penelope/logs/run_penelope_mat',IMAT,'.err'
        ELSE IF(IMAT.LT.100)THEN
          WRITE(FNAME,'(A,A,I2,A)') TRIM(MATDIR),'/mat',IMAT,'.mat'
          WRITE(RFILE,'(A,I2,A)')
     1      'output/penelope/raw/mat',IMAT,'.dat'
          WRITE(EFILE,'(A,I2,A)')
     1      'output/penelope/logs/run_penelope_mat',IMAT,'.err'
        ELSE
          WRITE(FNAME,'(A,A,I3,A)') TRIM(MATDIR),'/mat',IMAT,'.mat'
          WRITE(RFILE,'(A,I3,A)')
     1      'output/penelope/raw/mat',IMAT,'.dat'
          WRITE(EFILE,'(A,I3,A)')
     1      'output/penelope/logs/run_penelope_mat',IMAT,'.err'
        ENDIF
        PMFILE(1) = FNAME

C  ****  Seed the random number generator (same rule as GEANT4).
        ISEED1 = ISEEDB + IMAT
        ISEED2 = ISEEDB + IMAT

        EABS(1,1)=50.0D0
        EABS(2,1)=50.0D0
        EABS(3,1)=50.0D0
        C1(1)=0.01D0
        C2(1)=0.01D0
        WCC(1)=1.0D2
        WCR(1)=1.0D2

C  ****  Initialise PENELOPE for this material. Log to /dev/null.
        OPEN(7,FILE='/dev/null',STATUS='UNKNOWN')
        OPEN(26,FILE=EFILE,STATUS='REPLACE')
        CALL PEINIT(EMAX_GRID,1,7,1,PMFILE)
        IF(IRETRN.NE.0)THEN
          WRITE(6,*)'PEINIT failed for',FNAME,' IRETRN=',IRETRN
          CLOSE(7)
          GOTO 800
        ENDIF
        CLOSE(7)

C  ****  Open raw text output for this material.
        OPEN(10,FILE=RFILE,STATUS='REPLACE')
C  ****  First line: the run configuration (build_tables.py reads it).
        WRITE(10,1001) STEP_CM, EMIN_SIM, EMAX_SIM, NENRG, NHIST,
     1    ISEEDB, DEN(1)
 1001   FORMAT('# config: engine=penelope step_cm=',1PE15.8,
     1         ' emin_eV=',1PE15.8,' emax_eV=',1PE15.8,
     2         ' nenrg=',I0,' nhist=',I0,' seed=',I0,
     3         ' rho_gcm3=',1PE15.8)
        WRITE(10,1000)
 1000   FORMAT('# E0_keV theta_rad DE_total_eV dE_kin_eV',
     1         ' E_final_eV annihil_flag matID z_cm r_cm dphi_rad')

C  ****  Energy loop.
        DO IE = 1, NENRG
          E0 = ENERGY(IE)

          DO IH = 1, NHIST
C  ****  Initial state: positron at origin moving in +z.
            E = E0
            X = 0.0D0
            Y = 0.0D0
            Z = 0.0D0
            U = 0.0D0
            V = 0.0D0
            W = 1.0D0
            WGHT = 1.0D0
            KPAR = 3
            MAT = 1
            IBODY = 1
            DO J = 1, 5
              ILB(J) = 0
            ENDDO

            CALL CLEANS
            CALL START

            DSTOT = 0.0D0
            TOTDE = 0.0D0
            LANN = .FALSE.

            DO WHILE(DSTOT.LT.STEP_CM .AND. E.GT.0.0D0)
              DSMAX = STEP_CM - DSTOT
              CALL JUMP(DSMAX,DS)
C  ****  Move the positron (PENELOPE leaves that to the caller).
              X = X + U*DS
              Y = Y + V*DS
              Z = Z + W*DS
              DSTOT = DSTOT + DS
              CALL KNOCK(DEH,ICOL)
              TOTDE = TOTDE + DEH
              IF(E.LE.0.0D0)THEN
                LANN = .TRUE.
                EXIT
              ENDIF
            ENDDO

C  ****  Scattering angle and kinetic-energy loss.
            IF(W.GT.1.0D0) W = 1.0D0
            IF(W.LT.-1.0D0) W = -1.0D0
            THETA = ACOS(W)
            DEKIN = E0 - E
            IANN = 0
            IF(LANN) IANN = 1

C  ****  Azimuth of the final direction's transverse part (U,V) measured
C  ****  from the lateral displacement (X,Y), in (-pi, pi]; 0 when either
C  ****  vanishes. With theta, z and r it fixes the end state of the step up
C  ****  to a rotation about the initial direction.
            DPHI = 0.0D0
            IF(X*X+Y*Y.GT.0.0D0 .AND. U*U+V*V.GT.0.0D0)
     1        DPHI = ATAN2(X*V-Y*U, X*U+Y*V)
C  ****  Write raw text line. The end position is relative to the start
C  ****  of the step: Z along the initial direction, R perpendicular to it.
            WRITE(10,2000) E0*1.0D-3, THETA, TOTDE, DEKIN, E, IANN,
     1        IMAT, Z, SQRT(X*X+Y*Y), DPHI
 2000       FORMAT(F11.4,1X,F10.6,1X,E14.6,1X,E14.6,1X,E14.6,
     1             1X,I1,1X,I2,1X,F11.7,1X,F11.7,1X,F10.6)
          ENDDO
        ENDDO

        CLOSE(10)
C  ****  Keep the PENELOPE log only when it holds something.
        FLUSH(26)
        INQUIRE(UNIT=26,SIZE=ISZ)
        IF(ISZ.EQ.0)THEN
          CLOSE(26,STATUS='DELETE')
        ELSE
          CLOSE(26)
        ENDIF

        WRITE(6,*)'  Done material',IMAT
  800   CONTINUE
      ENDDO

      WRITE(6,*)'All done.'
      END PROGRAM

      SUBROUTINE penAbort()
      IMPLICIT NONE
      WRITE(6,*)'PENELOPE critical error - penAbort called'
      STOP 1
      END SUBROUTINE penAbort
