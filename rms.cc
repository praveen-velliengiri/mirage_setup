#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"
#include "mirage/transpiler/transpile.h"
#include "mirage/nki_transpiler/transpile.h"

using namespace mirage;

int main(int argc, char **argv) {

  kernel::Graph graph;
  kernel::DTensor X = graph.new_input(
      {8, 4096}, {4096, 1}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor W = graph.new_input(
      {4096, 6144}, {6144, 1}, type::DT_FLOAT16, layout::DmemRowMajor);

  {
    dim3 grid_dim = {64, 1, 1}, block_dim = {128, 1, 1};
    namespace tb = mirage::threadblock;
    tb::Graph bgraph(grid_dim, block_dim, 64, 64);
    tb::STensor bX = bgraph.new_input(X, {-1, -1, -1}, 1, layout::SmemRowMajor);
    tb::STensor bW = bgraph.new_input(W, {1, -1, -1}, 0, layout::SmemRowMajor);
    tb::STensor bM = bgraph.matmul(bX, bW);
    tb::STensor bAccX =
        bgraph.forloop_accum(bX, type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP);
    tb::STensor bAccM =
        bgraph.forloop_accum(bM, type::TB_FORLOOP_ACCUM_NO_RED_OP);
    tb::STensor bO = bgraph.div(bAccM, bAccX);
    bgraph.mark_output(bO, {1, -1, -1}, -1, type::TB_EPILOGUE_NONE);
    std::vector<kernel::DTensor> outputs = graph.customized({X, W}, bgraph);
    assert(outputs.size() == 1);
  }

/*
  transpiler::TranspilerConfig config;
  config.target_cc = 80;
  transpiler::TranspileResult codegen = transpiler::transpile(&graph, config, {{4096, 1}, {6144, 1}});
  std::cout << codegen.error_type << "\n";
  std::cout << codegen.code << "\n";
*/

{
  using namespace nki_transpiler;
  NKITranspilerConfig config;
  config.target_cc = 10;
  NKITranspileResult codegen = transpile(&graph, config);
  auto error = codegen.error_state.errors;
  if (!error.empty()) {
    for (auto &err : error)
      std::cout << err << "\n";
  } else {
    std::cout << codegen.code << "\n";
  }
}

/*
  search::GeneratorConfig config =
      search::GeneratorConfig::get_default_config();
  std::string checkpoint_file_name = "checkpoint_rms.json";

  search::KernelGraphGenerator gen(
      graph, config, checkpoint_file_name.data());
  gen.generate_kernel_graphs();
*/
  return 0;
}
