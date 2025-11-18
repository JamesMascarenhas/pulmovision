#-----------------------------------------------------------------------------
set(MODULE_NAME PulmoVision)

# Python files that define your module
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  # Optional: if you want a package structure, uncomment these when you have them
  # ${MODULE_NAME}Lib/__init__.py
  # ${MODULE_NAME}Lib/Logic.py
  # ${MODULE_NAME}Lib/Utils.py
)

# Non-code resources (icon + .ui)
set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
)

# Build the scripted module
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  # You can keep or drop tests; for a course project, dropping is fine
  # WITH_GENERIC_TESTS
)

# Optional: only if you actually write a unittest in PulmoVision.py
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
