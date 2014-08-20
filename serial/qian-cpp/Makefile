INC_EIGEN = -I /home/rehnd/src/eigen/

serialCN: serialCN.cpp CompCell.cpp
	g++ -o serialCN serialCN.cpp CompCell.cpp -O3 $(INC_EIGEN)

cleanSerial:
	rm -f serialCN CompCell.o



ompCN: ompCN.cpp CompCellomp.o
	g++ -o ompCN ompCN.cpp CompCellomp.o -O3 $(INC_EIGEN) -fopenmp

CompCellomp.o: CompCellomp.cpp CompCellomp.h
	g++ -c CompCellomp.cpp CompCellomp.h -O3 $(INC_EIGEN) -fopenmp

cleanOMP:
	rm -f ompCN CompCellomp.o



superluCN: superluCN.o ${SUPERLULIB} ${BLASLIB}
	g++ -fopenmp -O3 -D__OPENMP -DPRNTlevel=0 -DDEBUGlevel=0 superluCN.o CompCellsuperlu.cpp CompCellsuperlu.h \
		${INC_SUPERLU} $(INC_EIGEN) ${SUPERLULIB} ${BLASLIB} -lm -o $@

superluCN.o: superluCN.cpp CompCellsuperlu.cpp CompCellsuperlu.h
	g++ -fopenmp -O3 -D__OPENMP -DPRNTlevel=0 -DDEBUGlevel=0 ${INC_SUPERLU} $(INC_EIGEN) \
		-c superluCN.cpp CompCellsuperlu.cpp CompCellsuperlu.h

cleanSuperlu:
	rm -f superluCN superluCN.o


petscCN: petscCN.cpp CompCellpetsc.cpp CompCellpetsc.h chkopts
	${CLINKER} -o petscCN petscCN.cpp CompCellpetsc.cpp CompCellpetsc.h \
		-O3 -fopenmp $(INC_EIGEN) ${PETSC_LIB} ${INC_PETSC}

cleanPetsc:
	rm -f petscCN
