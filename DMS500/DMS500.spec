# -*- mode: python -*-

block_cipher = None


a = Analysis(['DMS500.py'],
             pathex=['E:\\Py3.6_Proje\\DMS500'],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          name='DMS500',
          debug=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True )
