GOOD_WARN = -std=c++11
NVCC = nvcc
CXX = g++

# The following should be adjusted to the target GPU architecture:
# -gencode arch=compute_50,code=sm_50

mode = release

ifeq ($(mode),release)
	CXXFLAGS = -I/opt/cuda-10.0/include/ -I/usr/local/hdf5/include/ -I../flann-master/src/cpp/ -pipe -std=c++11 -Wall -pedantic -DNDEBUG -O2 -mtune=native -march=native
	NVCCFLAGS = -std=c++11 -m64 -gencode arch=compute_50,code=sm_70 -Xptxas="-v"  -DNDEBUG
else
   mode = debug
	CXXFLAGS = -I/opt/cuda-10.0/include/ -I/usr/local/hdf5/include/ -I../flann-master/src/cpp/ -g -pipe -std=c++11 -Wall -O0 -Wcast-align  -Wctor-dtor-privacy -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op  -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls   -Wsign-promo -Wstrict-null-sentinel -Wstrict-overflow=5 -Wswitch-default -Wundef -Werror -Wno-unused
	NVCCFLAGS = -G -std=c++11 -m64 -gencode arch=compute_50,code=sm_70
endif

# This should point to a valid CUDA lib64/ location
LDFLAGS  = -L/opt/cuda-10.0/lib64/ -lcuda -lcudart -lcusparse -lcudadevrt -L/usr/local/lib/ -llz4 

BUILDDIR = obj
TARGET = gpuants
SRCDIR = src
SOURCES = main.cc ants.cc common.cc stopcondition.cc local_search.cc
CUDA_SOURCES = gpu_ants.cu  gpu_phmem.cu tsp_ls_gpu.cu cuda_utils.cu gpu_acs.cu

OBJS = $(SOURCES:.cc=.o)

CUDA_OBJS = $(CUDA_SOURCES:.cu=.o)

$(info $$OBJS is [${OBJS}])

OUT_OBJS = $(addprefix $(BUILDDIR)/,$(OBJS))
OUT_CUDA_OBJS = $(addprefix $(BUILDDIR)/,$(CUDA_OBJS))

DEPS = $(SOURCES:%.cc=$(BUILDDIR)/%.depends)
DEPS += $(SOURCES:%.cu=$(BUILDDIR)/%.depends)

$(warning $(DEPS))

.PHONY: clean all

all: $(TARGET)

$(TARGET): $(OUT_OBJS) $(OUT_CUDA_OBJS)
	$(NVCC) $(NVCCFLAGS) $(OUT_CUDA_OBJS) -dlink -o $(BUILDDIR)/link.o
	$(CXX) $(CXXFLAGS) $(OUT_OBJS) $(OUT_CUDA_OBJS) $(BUILDDIR)/link.o $(LDFLAGS) -o $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cc
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BUILDDIR)/%.depends: $(SRCDIR)/%.cc
	@mkdir -p depends
	$(CXX) -MF"$@" -MG -MM -MP  -MT"$(<F:%.cc=$(BUILDDIR)/%.o)" $(CXXFLAGS) $< > $@

$(BUILDDIR)/%.depends: $(SRCDIR)/%.cu
	$(CXX) -MF"$@" -MG -MM -MP  -MT"$(<F:%.cu=$(BUILDDIR)/%.o)" $(CXXFLAGS) $< > $@

clean:
	rm -f $(OUT_OBJS) $(OUT_CUDA_OBJS) $(DEPS) $(TARGET)

-include $(DEPS)
