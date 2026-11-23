C  *********************************************************************
C  decays_positron.f
C
C  Energy spectra of the particles released in the decay of one
C  radionuclide: the program that makes the beta+ emission spectra of
C  HMC (spectrum/generate_spectra.py).
C
C  Copyright (c) 2026 Universidad Complutense de Madrid, written for HMC
C  from the documented interface of PENNUC (CC BY-NC 4.0, as the rest of
C  spectrum/; see LICENSE of the repository). It
C  contains no PENNUC or PENELOPE code: it only calls the subroutines
C  PENNUC0 (initialisation) and PENNUC (one decay), reads the particles
C  they leave in the module PENNUC_mod, and seeds the random-number
C  generator of PENELOPE (RAND, common block /RSEED/). PENNUC and the
C  PENELOPE routines are not part of this file: build.sh compiles them
C  from your PenSuite / PENELOPE2023 installation (NEA Data Bank
C  package NEA-1525), under their own licence.
C
C  Input, standard input, one item per line:
C     1. name of the PENNUC data file (e.g. F-18.nuc), in the directory
C        that PENNUC_mod defines (build.sh sets it to nuclides/)
C     2. number of decays to simulate
C  Output, in the working directory:
C     spcnuc.dat  spectra of electrons, photons and positrons per decay
C                 and per eV, 400 bins of equal width between 0 and the
C                 highest energy of the emitted particles (EPMAX), with
C                 the statistical uncertainty (3 sigma) of each value
C     pennuc.dat  summary of the decay data, written by PENNUC0
C  Each nucleus counts as one history (all its cascades together), so
C  the uncertainties include the correlation between particles emitted
C  in the same decay. The seeds are fixed: the same file and number of
C  decays always give the same spectra.
C  *********************************************************************
      INCLUDE 'pennuc.f'
C
      PROGRAM DECAYS_POSITRON
      USE PENNUC_mod
      IMPLICIT NONE
      INTEGER*4, PARAMETER :: NB=400      ! energy bins
      INTEGER*4, PARAMETER :: NKIND=3     ! 1 e-, 2 photons, 3 e+
      INTEGER*4, PARAMETER :: NMAXP=5000  ! particles kept per nucleus
      INTEGER*4, PARAMETER :: IWR=7, IOUT=8
      DOUBLE PRECISION, PARAMETER :: FLOOR=1.0D-35
      CHARACTER NUCFNAME*45
      CHARACTER*8 KNAME(NKIND)
      INTEGER*8 NDEC, ID
      INTEGER*4 ISEED1, ISEED2
      INTEGER*4 IS, IER, METAST, J, L, K, IB, NP, NCOUNT, NLOST
      INTEGER*4 PK(NMAXP), PB(NMAXP)
      DOUBLE PRECISION EPMAX, DE, S1(NB,NKIND), S2(NB,NKIND)
      DOUBLE PRECISION XN, MEAN, VAR, VAL, UNC, T0, T1
      DOUBLE PRECISION V(NKIND), U(NKIND)
      COMMON/RSEED/ISEED1,ISEED2
C
      READ(*,'(A)') NUCFNAME
      NUCFNAME=ADJUSTL(NUCFNAME)
      READ(*,*) NDEC
      IF (NDEC.LT.1) STOP 'decays_positron: number of decays < 1'
C
      ISEED1=1
      ISEED2=1
      OPEN(IWR,FILE='pennuc.dat',STATUS='UNKNOWN')
      CALL PENNUC0(NUCFNAME,EPMAX,IWR,IER)
      CLOSE(IWR)
      IF (IER.NE.0) THEN
        WRITE(*,'(A,I4)') ' decays_positron: PENNUC0 failed, IER =',IER
        STOP 1
      ENDIF
      WRITE(*,'(A,1P,E12.6,A)') ' EPMAX =',EPMAX,' eV'
      DE=EPMAX/DBLE(NB)
C
      S1=0.0D0
      S2=0.0D0
      NLOST=0
      CALL CPU_TIME(T0)
      DO ID=1,NDEC
        NP=0
        DO
          IS=1      ! label of the nuclide (PENNUC may change it)
          METAST=0  ! PENNUC sets it to 1 at a metastable level
          CALL PENNUC(IS,METAST,IER)
          IF (IER.NE.0) THEN
            WRITE(*,'(A,I4)') ' decays_positron: PENNUC, IER =',IER
            STOP 1
          ENDIF
          DO J=1,NR
            K=KPNR(J)
            IF (K.GE.1 .AND. K.LE.NKIND) THEN
              IB=MIN(NB,MAX(1,INT(ENR(J)/DE)+1))
              IF (NP.LT.NMAXP) THEN
                NP=NP+1
                PK(NP)=K
                PB(NP)=IB
              ELSE
                NLOST=NLOST+1
              ENDIF
            ENDIF
          ENDDO
          IF (METAST.EQ.0) EXIT
        ENDDO
C       group the particles of this nucleus by (kind, bin)
        DO J=1,NP
          IF (PK(J).GT.0) THEN
            NCOUNT=1
            DO L=J+1,NP
              IF (PK(L).EQ.PK(J) .AND. PB(L).EQ.PB(J)) THEN
                NCOUNT=NCOUNT+1
                PK(L)=-PK(L)
              ENDIF
            ENDDO
            S1(PB(J),PK(J))=S1(PB(J),PK(J))+DBLE(NCOUNT)
            S2(PB(J),PK(J))=S2(PB(J),PK(J))+DBLE(NCOUNT)**2
          ENDIF
        ENDDO
      ENDDO
      CALL CPU_TIME(T1)
      IF (NLOST.GT.0) WRITE(*,'(A,I10,A)') ' WARNING:',NLOST,
     &  ' particles beyond the first 5000 of a nucleus were not counted'
C
      KNAME(1)='electron'
      KNAME(2)='photon  '
      KNAME(3)='positron'
      XN=DBLE(NDEC)
      OPEN(IOUT,FILE='spcnuc.dat',STATUS='UNKNOWN')
      WRITE(IOUT,'(A,A)') ' #  Results from decays_positron.',
     &  ' Radionuclide: '//TRIM(NUCFNAME)
      WRITE(IOUT,'(A,I12)') ' #  Number of simulated decays = ',NDEC
      WRITE(IOUT,'(A,1P,E10.3,A)') ' #  Total CPU simulation time = ',
     &  T1-T0,' seconds'
      WRITE(IOUT,'(A,1P,E10.3,A)') ' #  Simulation speed = ',
     &  XN/MAX(T1-T0,1.0D-9),' decays/second'
      WRITE(IOUT,'(A)') ' #'
      WRITE(IOUT,'(A)') ' #  Energy spectra of emitted particles'//
     &  '     (STU = statistical uncertainty)'
      WRITE(IOUT,'(A)') ' #  1st column: particle energy (eV)'
      WRITE(IOUT,'(A)') ' #  2nd-3rd columns: electron spectrum and'//
     &  ' STU (3 sigmas)'
      WRITE(IOUT,'(A)') ' #  4th-5th columns: photon spectrum and'//
     &  ' STU (3 sigmas)'
      WRITE(IOUT,'(A)') ' #  6th-7th columns: positron spectrum and'//
     &  ' STU (3 sigmas)'
      WRITE(IOUT,'(A)') ' #  Spectra are per decay and per eV.'
      DO IB=1,NB
        DO K=1,NKIND
          MEAN=S1(IB,K)/XN
          VAR=MAX(0.0D0,S2(IB,K)/XN-MEAN*MEAN)
          VAL=MEAN/DE
          UNC=3.0D0*SQRT(VAR/XN)/DE
          V(K)=MAX(VAL,FLOOR)
          U(K)=MAX(UNC,FLOOR)
        ENDDO
        WRITE(IOUT,'(2X,1P,E12.6,3(2X,E12.6,2X,E8.2))')
     &    (DBLE(IB)-0.5D0)*DE,(V(K),U(K),K=1,NKIND)
      ENDDO
      CLOSE(IOUT)
      WRITE(*,'(A,I12)') ' N = ',NDEC
      WRITE(*,'(A,1P,E10.3,A)') ' CPU time = ',T1-T0,' seconds'
      END
