# Contributing

Read [ARCHITECTURE.md](ARCHITECTURE.md) first — especially the invariants.

## Before you send a change

1. Run the regression suite. It needs an X display:

   ```
   Xvfb :99 &            # on a machine without a screen
   DISPLAY=:99 python _check.py
   ```

   It prints numbers as it goes and ends with `ВСЁ ХОРОШО` ("all good"). It
   reads sample textures from `test/`. Game textures are not distributed with
   the project (they belong to their publishers): put your own there under the
   names `_check.py` uses. The small ESRGAN fixtures (`test/esr_tiny_*`) are
   random weights and are included.
   The files it expects are `tx_m_ruins_trim_04.dds`, `tx_colony_stone02.dds`,
   `tx_ex_stronghold_wall12.dds`, `tx_leaves_03.dds` (Morrowind),
   `Vurt_PineBranches.dds`, `vurt_FFleavesG2.dds`, `vurt_FFleavesG2_n.dds`
   (from a foliage mod) and `ps_fix.dds` (the author's Photoshop lighting normal of `tx_ex_stronghold_wall12`). Without them the suite
   stops at the first texture load.

2. If you changed a signature that another module calls, raise `BUILD` in every
   module (the program refuses to start with mixed builds).

3. If you added or changed a node, re-run `python docs/make_node_docs.py`
   so that `docs/NODES.md` matches the program.

4. If you added a UI string, add it to `i18n.py` in both languages. The suite
   fails on a missing translation.

5. If you added a feature, make its effect visible — a view, a number under the
   sliders, or a 3D mode — and add a check to `_check.py` that measures it.

## What will not be accepted

- Anything that makes a map non-seamless (non-wrapping filters, gradients or
  padding in code that produces saved maps).
- Touching Tk from a worker thread.
- Reformatting or restructuring existing code without a functional reason.
  Keep diffs about what they change.

## Style

- Docstrings in English: purpose, parameters with units and ranges, return
  shape/dtype/range, and why the method is done this way when it is not
  obvious.
- Comments inside functions may stay in Russian — many of them explain
  decisions and are kept as they were written.

## AI-assisted work

Much of the code was written with an AI assistant. That is fine for
contributions too; say so with an `Assisted-by:` line in the commit message.
The person sending the change is responsible for it.
