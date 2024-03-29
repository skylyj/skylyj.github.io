# Emacs配置文件


## 基础配置 {#基础配置}


### straight {#straight}

```emacs-lisp
(defvar bootstrap-version)
(let ((bootstrap-file
       (expand-file-name "straight/repos/straight.el/bootstrap.el" user-emacs-directory))
      (bootstrap-version 5))
  (unless (file-exists-p bootstrap-file)
    (with-current-buffer
        (url-retrieve-synchronously
         "https://raw.githubusercontent.com/raxod502/straight.el/develop/install.el"
         'silent 'inhibit-cookies)
      (goto-char (point-max))
      (eval-print-last-sexp)))
  (load bootstrap-file nil 'nomessage))
```


### use package {#use-package}

```emacs-lisp
(require 'package)
(add-to-list 'package-archives '("gnu"   . "https://elpa.gnu.org/packages/"))
(add-to-list 'package-archives '("melpa" . "https://melpa.org/packages/"))
(package-initialize)
(unless (package-installed-p 'use-package)
  (package-refresh-contents)
  (package-install 'use-package))
(eval-and-compile
  (setq use-package-always-ensure t
        use-package-expand-minimally t))
```


### dired {#dired}

```emacs-lisp
(global-set-key (kbd "C-x C-j") 'dired-jump)
(require 'dired-x)
(setq dired-omit-files
      (concat dired-omit-files "\\|^\\.DS_Store$" "\\|^\\..+$" "\\|__pycache__" "\\|.*\\.pyc$"))
(add-hook 'dired-mode-hook 'dired-omit-mode)
(use-package dired-sidebar
  :bind (("C-x C-n" . dired-sidebar-toggle-sidebar))
  :ensure t
  :commands (dired-sidebar-toggle-sidebar)
  :init
  (add-hook 'dired-sidebar-mode-hook
            (lambda ()
              (unless (file-remote-p default-directory)
                (auto-revert-mode))))
  :config
  (push 'toggle-window-split dired-sidebar-toggle-hidden-commands)
  (push 'rotate-windows dired-sidebar-toggle-hidden-commands)
  (setq dired-sidebar-subtree-line-prefix "__")
  (setq dired-sidebar-theme 'vscode)
  (setq dired-sidebar-use-term-integration t)
  (setq dired-sidebar-use-custom-font t))
(use-package dired-subtree
  :ensure t
  :defer t
  :bind (:map dired-mode-map
              ("TAB" . dired-subtree-cycle)))
(use-package dired-collapse
  :ensure t
  :config
  (add-hook 'dired-mode-hook 'dired-collapse-mode)
  )
(use-package dired-rainbow
  :config
  (progn
    (dired-rainbow-define-chmod directory "#6cb2eb" "d.*")
    (dired-rainbow-define html "#eb5286" ("css" "less" "sass" "scss" "htm" "html" "jhtm" "mht" "eml" "mustache" "xhtml"))
    (dired-rainbow-define xml "#f2d024" ("xml" "xsd" "xsl" "xslt" "wsdl" "bib" "json" "msg" "pgn" "rss" "yaml" "yml" "rdata"))
    (dired-rainbow-define document "#9561e2" ("docm" "doc" "docx" "odb" "odt" "pdb" "pdf" "ps" "rtf" "djvu" "epub" "odp" "ppt" "pptx"))
    (dired-rainbow-define markdown "#ffed4a" ("org" "etx" "info" "markdown" "md" "mkd" "nfo" "pod" "rst" "tex" "textfile" "txt"))
    (dired-rainbow-define database "#6574cd" ("xlsx" "xls" "csv" "accdb" "db" "mdb" "sqlite" "nc"))
    (dired-rainbow-define media "#de751f" ("mp3" "mp4" "MP3" "MP4" "avi" "mpeg" "mpg" "flv" "ogg" "mov" "mid" "midi" "wav" "aiff" "flac"))
    (dired-rainbow-define image "#f66d9b" ("tiff" "tif" "cdr" "gif" "ico" "jpeg" "jpg" "png" "psd" "eps" "svg"))
    (dired-rainbow-define log "#c17d11" ("log"))
    (dired-rainbow-define shell "#f6993f" ("awk" "bash" "bat" "sed" "sh" "zsh" "vim"))
    (dired-rainbow-define interpreted "#38c172" ("py" "ipynb" "rb" "pl" "t" "msql" "mysql" "pgsql" "sql" "r" "clj" "cljs" "scala" "js"))
    (dired-rainbow-define compiled "#4dc0b5" ("asm" "cl" "lisp" "el" "c" "h" "c++" "h++" "hpp" "hxx" "m" "cc" "cs" "cp" "cpp" "go" "f" "for" "ftn" "f90" "f95" "f03" "f08" "s" "rs" "hi" "hs" "pyc" ".java"))
    (dired-rainbow-define executable "#8cc4ff" ("exe" "msi"))
    (dired-rainbow-define compressed "#51d88a" ("7z" "zip" "bz2" "tgz" "txz" "gz" "xz" "z" "Z" "jar" "war" "ear" "rar" "sar" "xpi" "apk" "xz" "tar"))
    (dired-rainbow-define packaged "#faad63" ("deb" "rpm" "apk" "jad" "jar" "cab" "pak" "pk3" "vdf" "vpk" "bsp"))
    (dired-rainbow-define encrypted "#ffed4a" ("gpg" "pgp" "asc" "bfe" "enc" "signature" "sig" "p12" "pem"))
    (dired-rainbow-define fonts "#6cb2eb" ("afm" "fon" "fnt" "pfb" "pfm" "ttf" "otf"))
    (dired-rainbow-define partition "#e3342f" ("dmg" "iso" "bin" "nrg" "qcow" "toast" "vcd" "vmdk" "bak"))
    (dired-rainbow-define vc "#0074d9" ("git" "gitignore" "gitattributes" "gitmodules"))
    (dired-rainbow-define-chmod executable-unix "#38c172" "-.*x.*")
    ))
(use-package dired-filter
  :ensure t)
(use-package dired-narrow
  :ensure t
  :bind (:map dired-mode-map
              ("," . dired-narrow))
  )
(use-package dired-ranger
  :ensure t)
```


### other {#other}

```emacs-lisp
  (setq make-backup-files nil)
  (setq ns-pop-up-frames nil)
  ;; (global-set-key [remap goto-line] 'goto-line-preview)
  ;; (setq eshell-last-dir-ring-size 500)
  (add-hook 'sh-mode-hook 'flycheck-mode)
  (if (display-graphic-p)
      (progn
        (scroll-bar-mode -1)
        (load-theme 'misterioso)
        )
    (progn
      (menu-bar-mode -1)
      ;; (toggle-scroll-bar -1)
      (tool-bar-mode -1)
                                          ;(load-theme 'zenburn)
      )
    )
  ;; for mac
  (setq mac-command-modifier 'meta)
  (setq mac-option-modifier 'super)
  (setq ns-function-modifier 'hyper)  ; make Fn key do Hyper
  (setq abbrev-file-name "~/.emacs.d/personal/.abbrev_defs") ;; 缺省的定义缩写的文件。
  (setq my/savefile-dir "~/.emacs.d/personal/")
  (setq bookmark-default-file (expand-file-name "bookmarks" my/savefile-dir)) ;; 缺省书签文件的路径及文件名。
  ;; save recent files
  (require 'recentf)
  (setq recentf-save-file (expand-file-name "recentf" my/savefile-dir)
        recentf-max-saved-items 500
        recentf-max-menu-items 15
        recentf-auto-cleanup 'never)
  (global-font-lock-mode 1)               ; 开启语法高亮。
  (auto-compression-mode 1)               ; 打开压缩文件时自动解压缩。
  (column-number-mode 1)                  ; 显示列号。
  (blink-cursor-mode -1)                  ; 光标不要闪烁。
  ;; (display-time-mode 1)                   ; 显示时间。
  (show-paren-mode 1)                     ; 高亮显示匹配的括号。
  (icomplete-mode 1)            ; 给出用 M-x foo-bar-COMMAND 输入命令的提示。
  (setq select-enable-clipboard t)  ;用来和系统共享剪贴板
  (setq confirm-kill-emacs 'yes-or-no-p)
  (transient-mark-mode t)
  (set-face-attribute 'region nil :background "#666" :foreground "#ffffff")
  ;; 可以让选择的区域高亮
  (line-number-mode t)
  (setq-default kill-whole-line t) ;; 在行首 C-k 时，同时删除该行。
  (fset 'yes-or-no-p 'y-or-n-p) ;;改变 Emacs 固执的要你回答 yes 的行为。按y或空格键表示yes,n表示no。
  (setq auto-image-file-mode t)
  ;; 系统相关
(set-frame-font "Menlo-16" nil t)
  (cond
   ((equal system-type 'gnu/linux)
    (set-frame-font "Monospace-19"))
   ((equal system-type 'darwin)
    (tool-bar-mode -1)
    (set-frame-font "Menlo-16")
    (add-to-list 'default-frame-alist
             '(font . "Menlo-16"))
    ;; (set-fontset-font
    ;;  (frame-parameter nil 'font)
    ;;  'han
    ;;  (font-spec :family "Hiragino Sans GB" ))
    ))
  (setq auto-save-default nil)
  (setq whitespace-style '(tabs trailing lines tab-mark))
  (use-package exec-path-from-shell
    :ensure t
    :config
    (when (memq window-system '(mac ns x))
      (exec-path-from-shell-initialize))
    )
  ;; (provide 'basic_conf)
  (add-hook 'dired-after-readin-hook 'hl-line-mode)
  (require 'uniquify)
  (setq uniquify-buffer-name-style 'reverse)
  (setq uniquify-separator "/")
  (setq uniquify-after-kill-buffer-p t) ; rename after killing uniquified
  (setq uniquify-ignore-buffers-re "^\\*") ; don't muck with special buffers
  (when (display-graphic-p)
    (use-package csv-mode
      :ensure t
      :config
      (add-hook 'csv-mode-hook 'csv-align-mode))
                                          ;(use-package vterm
                                          ;  :ensure t)
                                          ;(use-package multi-vterm
                                          ;  :ensure t)
    )
  (add-to-list 'package-archives '("melpa-stable" . "https://stable.melpa.org/packages/") t)
```


### doc and pdf {#doc-and-pdf}

```emacs-lisp
(use-package pdf-tools
:ensure t
:config
(pdf-tools-install)
;; automatically open pdfs with pdf-tools
(setq-default pdf-view-display-size 'fit-page)
(add-to-list 'auto-mode-alist '("\\.pdf\\'" . pdf-view-mode)))

  (add-hook 'pdf-view-mode-hook (lambda () (display-line-numbers-mode -1)))
  (use-package org-noter
  :ensure t )
  (use-package devdocs
    :ensure t
    :config
    (global-set-key (kbd "C-h D") 'devdocs-lookup)
    )
  (use-package devdocs-browser
    :ensure t)
```


### emacs server {#emacs-server}

```emacs-lisp
;; emacs server, client 可以在终端使用 /Applications/Emacs\ 2.app/Contents/MacOS/bin/emacsclient ifuns.el &来启动
(when (display-graphic-p)
  (server-start)
  )
```


### terminal {#terminal}

```emacs-lisp
(use-package vterm
  :when (memq window-system '(mac ns x pgtk))
  :bind (:map vterm-mode-map
              ("C-y" . vterm-yank)
              ("M-y" . vterm-yank-pop)
              ("C-k" . vterm-send-C-k-and-kill))
  :init
  (setq vterm-shell "zsh")
  :config
  (setq vterm-always-compile-module t)
  (setq vterm-buffer-name-string "vterm %s")
  (defun vterm-send-C-k-and-kill ()
    "Send `C-k' to libvterm, and put content in kill-ring."
    (interactive)
    (kill-ring-save (point) (vterm-end-of-line))
    (vterm-send-key "k" nil nil t)))
(use-package vterm-toggle
  :when (memq window-system '(mac ns x pgtk))
  ;; :bind (([f8] . vterm-toggle)
  ;;        ([f9] . vterm-compile)
  ;;        :map vterm-mode-map
  ;;        ([f8] . vterm-toggle)
  ;;        ([(control return)] . vterm-toggle-insert-cd))
  :config
  (setq vterm-toggle-cd-auto-create-buffer nil)
  (defvar vterm-compile-buffer nil)
  (defun vterm-compile ()
    "Compile the program including the current buffer in `vterm'."
    (interactive)
    (setq compile-command (compilation-read-command compile-command))
    (let ((vterm-toggle-use-dedicated-buffer t)
          (vterm-toggle--vterm-dedicated-buffer (if (vterm-toggle--get-window)
                                                    (vterm-toggle-hide)
                                                  vterm-compile-buffer)))
      (with-current-buffer (vterm-toggle-cd)
        (setq vterm-compile-buffer (current-buffer))
        (rename-buffer "*vterm compilation*")
        (compilation-shell-minor-mode 1)
        (vterm-send-M-w)
        (vterm-send-string compile-command t)
        (vterm-send-return)))))
```


## 项目管理 {#项目管理}


### magit {#magit}

```emacs-lisp
(use-package magit
  :ensure t
  )
(use-package magit-delta
  :hook (magit-mode . magit-delta-mode))
```


## edit {#edit}


### 光标移动 {#光标移动}

```emacs-lisp
  (use-package smartscan
    :ensure t
    :config
    (add-hook 'prog-mode-hook 'smartscan-mode)
    (add-hook 'org-mode-hook 'smartscan-mode)
    (add-hook 'dired-mode-hook 'smartscan-mode)
    )
  ;; enable a more powerful jump back function from ace jump mode
  (use-package ace-jump-mode
    :ensure t
    :config
    (define-key global-map (kbd "C-c SPC") 'ace-jump-mode)
    (autoload
      'ace-jump-mode-pop-mark
      "ace-jump-mode"
      "Ace jump back:-)"
      t)
    (eval-after-load "ace-jump-mode"
      '(ace-jump-mode-enable-mark-sync))
    )

(use-package ace-pinyin
    :ensure t
    :config
    (setq ace-pinyin-use-avy nil)
    (ace-pinyin-global-mode +1)
    (global-set-key (kbd "s-i pj") 'ace-pinyin-jump-char)
    )
```


### 选择和拷贝 {#选择和拷贝}

```emacs-lisp
(use-package expand-region
  :bind ("C-=" . er/expand-region))
;; alt-w b 拷贝当前文件的路径
(use-package easy-kill
  :ensure t
  :config
  (global-set-key [remap kill-ring-save] 'easy-kill)
  (global-set-key [remap mark-sexp] 'easy-mark)
  )
```


### bookmark {#bookmark}


#### bm {#bm}

```emacs-lisp
(use-package bm
  :ensure t
  :demand t
  :init
  (setq bm-restore-repository-on-load t)
  :config
  (setq bm-cycle-all-buffers t)
  (setq bm-repository-file "~/.emacs.d/bm-repository")
  (setq-default bm-buffer-persistence t)
  (add-hook 'after-init-hook 'bm-repository-load)
  (add-hook 'kill-buffer-hook #'bm-buffer-save)
  (add-hook 'kill-emacs-hook #'(lambda nil
                                 (bm-buffer-save-all)
                                 (bm-repository-save)))
  (add-hook 'after-save-hook #'bm-buffer-save)
  (add-hook 'find-file-hooks   #'bm-buffer-restore)
  (add-hook 'after-revert-hook #'bm-buffer-restore)
  (add-hook 'vc-before-checkin-hook #'bm-buffer-save)
  (setq bm-marker 'bm-marker-right)
  (global-set-key (kbd "<left-fringe> <M-mouse-1>") 'bm-toggle-mouse)
  (global-set-key (kbd "s-i bn") 'bm-next)
  (global-set-key (kbd "s-i bp") 'bm-previous)
  (global-set-key (kbd "s-i bt") 'bm-toggle)
  (global-set-key (kbd "s-i bl") 'bm-show-all)
  )
```


#### bookmark-int-project {#bookmark-int-project}

```emacs-lisp
  (use-package bookmark-in-project
    :ensure t
    :commands (bookmark-in-project-jump
               bookmark-in-project-jump-next
b               bookmark-in-project-jump-previous
               bookmark-in-project-delete-all)
    :bind (
           ;; ("s-i pbl" . bookmark-in-project-jump)
           ("s-i pbn" . bookmark-in-project-jump-next)
           ("s-i pbt" . bookmark-in-project-toggle)
           ("s-i pbp" . bookmark-in-project-jump-previous)
           ("s-i pbd" . bookmark-in-project-delete-all)
           ))
  (load-library "bookmark-in-project")
  (defun my/bookmark-in-project-jump ()
    "Jump to a bookmark in the current project."
    (interactive)
    (bookmark-maybe-load-default-file)
    (bookmark-in-project--jump-impl #'bookmark-jump))
  (global-set-key (kbd "s-i pbl") 'my/bookmark-in-project-jump)
```


### 配对的括号编辑 {#配对的括号编辑}


#### change-inner {#change-inner}

```emacs-lisp
(use-package change-inner
  :ensure t
  :config
  :bind (("s-i ci" . change-inner)
         ("s-i co" . change-outer)
         )
  )
```


#### embrace {#embrace}

```emacs-lisp
;; 删除修改配对的括号，这里的括号包括：(),[],{},<>,单引号，双引号
(use-package embrace
  :ensure t
  :bind (("s-i ," . embrace-commander))
  )
```


### basic {#basic}

```emacs-lisp
(when (display-graphic-p)
  ;; Do any keybindings and theme setup here
  (use-package gnuplot
    :ensure t)
  (use-package gnuplot-mode
    :ensure t)
  (use-package helm-chrome
    :ensure t)
  (use-package diminish
    :ensure t)
  (use-package visual-regexp
    :ensure t)
  (use-package visual-regexp-steroids
    :ensure t
    :config
    (define-key global-map (kbd "C-c q") 'vr/query-replace)
    )
  ;; for forge
                                        ;(setq auth-sources '("~/.authinfo"))
                                        ;(use-package forge
                                        ;  :ensure t
                                        ;  )
                                        ;(with-eval-after-load 'magit
                                        ;  (require 'forge))
                                        ;(with-eval-after-load 'forge
                                        ;  (add-to-list 'forge-alist
                                        ;               '("gitlab.mobvista.com"
                                        ;                 "gitlab.mobvista.com/api/v4"
                                        ;                 "gitlab.mobvista.com"
                                        ;                 forge-gitlab-repository)))
  ;;google this
  (use-package google-this
    :ensure t
    :config
    (google-this-mode 1)
    (global-set-key (kbd "C-c g") 'google-this-mode-submap)
    )
  ;; youtube
  (use-package helm-youtube
    :ensure t
    :config
    (setq helm-youtube-key "AIzaSyAQB8odOXYv46YR-x0Dk7BZbVTnWVYL4oM")
    (define-key global-map (kbd "C-c y") 'helm-youtube)
    )
  (use-package good-scroll
    :ensure t
    :config
    (good-scroll-mode -1)
    )
  (use-package beacon
    :ensure t
    :config
    (beacon-mode -1)
    )
  (use-package key-chord
    :ensure t
    :config
    (key-chord-mode 1)
    (key-chord-define-global "hj"     'undo)
    )
  (use-package which-key
    :ensure t
    :config
    (which-key-mode)
    )
  (use-package zop-to-char
    :ensure t
    :config
    (global-set-key (kbd "M-z") 'zop-up-to-char)
    (global-set-key (kbd "M-Z") 'zop-to-char)
    )
  (setq package-check-signature nil)
  (use-package undo-tree
    :ensure t
    :config
    (undo-tree-mode)
    (setq undo-tree-history-directory-alist
          `((".*" . ,temporary-file-directory)))
    (setq undo-tree-auto-save-history t)
    (global-undo-tree-mode)
    )
  (use-package crux
    :ensure t)
  (use-package anzu
    :ensure t
    :config
    (global-anzu-mode)
    (global-set-key (kbd "M-%") 'anzu-query-replace)
    (global-set-key (kbd "C-M-%") 'anzu-query-replace-regexp)
    )
  (use-package zenburn-theme
    :ensure t
    ;; :config
    ;; (load-theme 'zenburn t)
    )
  ;; (use-package openwith
  ;;   :ensure t
  ;;   :config
  ;;   (when (require 'openwith nil 'noerror)
  ;;     (setq openwith-associations
  ;;           (list
  ;;            (list (openwith-make-extension-regexp
  ;;                   '("doc" "docx" "xls" "ppt" "pptx" "docx" "pdf"))
  ;;                  "open"
  ;;                  '(file))
  ;;            (list (openwith-make-extension-regexp
  ;;                   '("pdf" "ps" "ps.gz" "dvi"))
  ;;                  "open"
  ;;                  '(file))
  ;;            ))
  ;;     (openwith-mode 1))
  ;;   )
  )
(use-package browse-kill-ring
  :disabled
  :ensure t
  :config
  (browse-kill-ring-default-keybindings)
  (global-set-key (kbd "s-y") 'browse-kill-ring)
  ;; 有了helm-show-kill-ring 就够用了
  )
(use-package transpose-frame
  :ensure t
  )
(display-line-numbers-mode)
(use-package which-key
  :config
  (which-key-mode))
;; ace-pinyin and evil-numbers 可能不兼容copilot
;; (use-package evil-numbers
;;   :ensure t
;;   :config
;;   (global-set-key (kbd "C-c +") 'evil-numbers/inc-at-pt)
;;   (global-set-key (kbd "C-c -") 'evil-numbers/dec-at-pt)
;;   (global-set-key (kbd "C-c C-+") 'evil-numbers/inc-at-pt-incremental)
;;   (global-set-key (kbd "C-c C--") 'evil-numbers/dec-at-pt-incremental)
;;   )
```


## appearance {#appearance}


### theme {#theme}


#### doom-themes {#doom-themes}

```emacs-lisp
(use-package doom-themes
  :ensure t
  :config
  ;; Global settings (defaults)
  (setq doom-themes-enable-bold t    ; if nil, bold is universally disabled
        doom-themes-enable-italic t) ; if nil, italics is universally disabled
  ;; (load-theme 'doom-zenburn t)
  ;; (load-theme 'doom-snazzy t)
  (load-theme 'doom-zenburn t)
  (doom-themes-visual-bell-config)
  (doom-themes-neotree-config)
  (setq doom-themes-treemacs-theme "doom-atom") ; use "doom-colors" for less minimal icon theme
  (doom-themes-treemacs-config)
  (doom-themes-org-config))
```


#### helm-themes {#helm-themes}

```emacs-lisp
(use-package helm-themes
  :ensure t
  )
```


### window {#window}

```emacs-lisp
(use-package ace-window
  :ensure t
  :bind (("C-x o" . ace-window)))
(use-package workgroups2
  :ensure t
  :init
  (setq wg-prefix-key "C-c w")
  (setq wg-session-file "~/.emacs.d/.emacs_workgroups")
  :config
  (workgroups-mode 1))
(winner-mode 1)
```


### rainbow {#rainbow}

```emacs-lisp
(use-package rainbow-delimiters
  :ensure t
  :config
  (add-hook 'prog-mode-hook 'rainbow-delimiters-mode)
  (add-hook 'org-mode-hook 'rainbow-delimiters-mode)
  )
```


### doom-modeline {#doom-modeline}

```emacs-lisp
(prefer-coding-system 'utf-8)
(set-default-coding-systems 'utf-8)
(set-terminal-coding-system 'utf-8)
(set-keyboard-coding-system 'utf-8)
(setq default-buffer-file-coding-system 'utf-8)
(setq system-time-locale "zh_CN.UTF-8")
  (when (display-graphic-p)
    ;; Do any keybindings and theme setup here
    (setenv "JAVA_HOME" "/Library/Java/JavaVirtualMachines/adoptopenjdk-8.jdk/Contents/Home")
    (use-package all-the-icons
      :ensure t)
    (use-package all-the-icons-ivy-rich
      :ensure t
      :config
      (all-the-icons-ivy-rich-mode 1)
      )
    (use-package all-the-icons-dired
      :ensure t
      :config
      (add-hook 'dired-mode-hook 'all-the-icons-dired-mode)
      )
    (use-package doom-modeline
      :ensure t
      :hook (after-init . doom-modeline-mode)
      :config
      (setq doom-modeline-env-version t)
      (setq doom-modeline-env-enable-python t)
      (setq doom-modeline-env-enable-ruby t)
      (setq doom-modeline-env-enable-perl t)
      (setq doom-modeline-env-enable-go t)
      (setq doom-modeline-env-enable-elixir t)
      (setq doom-modeline-env-enable-rust t)
      (setq doom-modeline-buffer-encoding t)
      (setq doom-modeline-workspace-name t)
      (setq doom-modeline-buffer-file-name-style 'buffer-name)
      :custom-face
      (mode-line ((t (:height 1.0))))
      (mode-line-inactive ((t (:height 1.0))))
      :custom
      (doom-modeline-height 18)
      (doom-modeline-bar-width 10)
      (doom-modeline-lsp t)
      (doom-modeline-github t)
      (doom-modeline-mu4e nil)
      (doom-modeline-irc t)
      (doom-modeline-buffer-encoding t)
      (doom-modeline-battery t)
      ;; (doom-modeline-minor-modes t)
      ;; (doom-modeline-persp-name nil)
      (doom-modeline-major-mode-icon t)
      )
    (with-eval-after-load 'sr-speedbar
      (add-hook 'speedbar-visiting-file-hook
                #'(lambda () (select-window (next-window))) t))
    ;; (use-package neotree
    ;;   :ensure t
    ;;   )
    )
```


## program {#program}


### appearance {#appearance}


#### 边栏 {#边栏}

```emacs-lisp
(when (display-graphic-p)
  (use-package treemacs
    :ensure t
    :bind
    (:map global-map
          ([f8]       . treemacs-select-window)
          )
    )
  )
```


#### code folding {#code-folding}

```emacs-lisp
;;hide and show
(add-hook 'emacs-lisp-mode-hook 'hs-minor-mode)
(add-hook 'python-mode-hook 'hs-minor-mode)
(add-hook 'go-mode 'hs-minor-mode)
(add-hook 'c++-mode 'hs-minor-mode)
(add-hook 'prog-mode-hook 'outline-minor-mode)
(add-hook 'prog-mode-hook 'hs-minor-mode)
;; (use-package bicycle
;;   :bind (:map outline-minor-mode-map
;;               ([C-tab] . bicycle-cycle)
;;               ([S-tab] . bicycle-cycle-global))
;;   )
(use-package yafolding
  :bind (:map yafolding-mode-map
              ([C-tab] . yafolding-toggle-element)
              ([S-tab] . yafolding-toggle-all))
  :config
  (add-hook 'prog-mode-hook 'yafolding-mode)
  )
(use-package outline-minor-faces
  :after outline
  :config
  (add-hook 'outline-minor-mode-hook #'outline-minor-faces-mode))
```


### documentation {#documentation}

```emacs-lisp
(use-package dash-at-point
  :ensure t
  :config
  (global-set-key (kbd "s-i dp") 'dash-at-point)
  (global-set-key (kbd "s-i dw") 'dash-at-point-with-docset)
  )
```


### 补全 {#补全}

```emacs-lisp
(use-package company
  :ensure t
  :hook (scala-mode . company-mode)
  :config
  (add-hook 'after-init-hook 'global-company-mode)
  (define-key company-search-map (kbd "C-n") 'company-select-next)
  (define-key company-search-map (kbd "C-p") 'company-select-previous)
  (define-key company-active-map (kbd "M-<") 'company-select-first)
  (define-key company-active-map (kbd "M->") 'company-select-last)
  (setq lsp-completion-provider :capf)
  :custom
  (company-minimum-prefix-length 3)
  (company-idle-delay 0.0))
(use-package copilot
  :straight (:host github :repo "zerolfx/copilot.el" :files ("dist" "*.el"))
  :ensure t
  :config
  (setq copilot-network-proxy '(:host "127.0.0.1" :port 61491))
  )
(add-hook 'prog-mode-hook 'copilot-mode)
(with-eval-after-load 'company
  (delq 'company-preview-if-just-one-frontend company-frontends))
(define-key copilot-completion-map (kbd "<tab>") 'copilot-accept-completion)
(define-key copilot-completion-map (kbd "TAB") 'copilot-accept-completion)
```


### helm {#helm}

```emacs-lisp
;; helm
(use-package helm
  :ensure t
  :config
  (setq helm-locate-fuzzy-match nil)
  (setq helm-locate-command "mdfind -name %s %s")
  (setq locate-command "mdfind")
  ;; (setq helm-follow-mode-persistent t)
  (global-set-key (kbd "C-c h") 'helm-command-prefix)
  (global-set-key (kbd "M-x") 'helm-M-x)
  (global-set-key (kbd "M-y") 'helm-show-kill-ring)
  (global-set-key (kbd "C-x b") 'helm-mini)
  (global-set-key (kbd "C-x C-b") 'helm-buffers-list)
  (global-set-key (kbd "C-x C-f") 'helm-find-files)
  (global-set-key (kbd "C-x r b") #'helm-filtered-bookmarks)
  (set-face-attribute 'helm-selection nil
                      :background "lime green"
                      :foreground "black")
  (setq
   helm-split-window-in-side-p           t
                                        ; open helm buffer inside current window,
                                        ; not occupy whole other window
   helm-move-to-line-cycle-in-source     t
                                        ; move to end or beginning of source when
                                        ; reaching top or bottom of source.
   helm-ff-search-library-in-sexp        t
                                        ; search for library in `require' and `declare-function' sexp.
   helm-scroll-amount                    8
                                        ; scroll 8 lines other window using M-<next>/M-<prior>
   helm-ff-file-name-history-use-recentf t
   ;; Allow fuzzy matches in helm semantic
   helm-semantic-fuzzy-match t
   helm-imenu-fuzzy-match    t)
  ;; Have helm automaticaly resize the window
  (helm-autoresize-mode 1)
  (setq rtags-use-helm t)
  (setq helm-buffer-max-length nil)
  )
(use-package helm-projectile
  :ensure t
  :config
  (helm-projectile-on)
  (setq projectile-completion-system 'helm)
  )
(use-package helm-swoop
  :ensure t
  :config
  (global-set-key (kbd "M-i") 'helm-swoop)
  (global-set-key (kbd "C-c M-i") 'helm-multi-swoop)
  (global-set-key (kbd "C-x M-i") 'helm-multi-swoop-all)
  ;; When doing isearch, hand the word over to helm-swoop
  (define-key isearch-mode-map (kbd "M-i") 'helm-swoop-from-isearch)
  ;; From helm-swoop to helm-multi-swoop-all
  (define-key helm-swoop-map (kbd "M-i") 'helm-multi-swoop-all-from-helm-swoop)
  ;; When doing evil-search, hand the word over to helm-swoop
  ;; (define-key evil-motion-state-map (kbd "M-i") 'helm-swoop-from-evil-search)
  ;; Instead of helm-multi-swoop-all, you can also use helm-multi-swoop-current-mode
  (define-key helm-swoop-map (kbd "M-m") 'helm-multi-swoop-current-mode-from-helm-swoop)
  ;; Move up and down like isearch
  (define-key helm-swoop-map (kbd "C-r") 'helm-previous-line)
  (define-key helm-swoop-map (kbd "C-s") 'helm-next-line)
  (define-key helm-multi-swoop-map (kbd "C-r") 'helm-previous-line)
  (define-key helm-multi-swoop-map (kbd "C-s") 'helm-next-line)
  )
(use-package helm-ag
  :ensure t)
(defun my-helm-ag-thing-at-point ()
  "Search the symbol at point with `helm-ag'."
  (interactive)
  (let ((helm-ag-insert-at-point 'symbol))
    (helm-projectile-ag)
    ;; (helm-do-ag-project-root)
    ))
(global-set-key (kbd "M-I") 'my-helm-ag-thing-at-point)
(use-package helm-ls-git
  :ensure t
  )
(use-package helm-ctest
  :ensure t
  )
(use-package helm-flycheck
  :ensure t
  :config
  (eval-after-load 'flycheck
    '(define-key flycheck-mode-map (kbd "C-c ! h") 'helm-flycheck))
  )
;; (require 'swiper-helm)
(use-package helm-bm
  :ensure t
  )
```


### lsp {#lsp}

```emacs-lisp
    ;; lsp related
    (use-package lsp-mode
      :ensure t
      :hook
      (scala-mode . lsp)
      (lsp-mode . lsp-lens-mode)
      (python-mode . lsp)
      (c++-mode . lsp)
      (sh-mode . lsp-deferred)
      (yaml-mode . lsp)
      :init
      (setq lsp-bash-server-command '("bash-language-server" "start"))
      :config
      ;; Uncomment following section if you would like to tune lsp-mode performance according to
      ;; https://emacs-lsp.github.io/lsp-mode/page/performance/
      ;; (setq gc-cons-threshold 100000000) ;; 100mb
      ;; (setq read-process-output-max (* 1024 1024)) ;; 1mb
      ;; (setq lsp-idle-delay 0.500)
      ;; (setq lsp-log-io nil)
      ;; (setq lsp-completion-provider :capf)
      (setq lsp-prefer-flymake nil)
      ;; Makes LSP shutdown the metals server when all buffers in the project are closed.
      ;; https://emacs-lsp.github.io/lsp-mode/page/settings/mode/#lsp-keep-workspace-alive
      (setq lsp-keep-workspace-alive nil))
  (with-eval-after-load 'flycheck
    (add-to-list 'flycheck-disabled-checkers 'python-flake8)
    (flycheck-add-next-checker 'python-pylint
                               '(warning . python-flake8)))
  ;; Disable pylint for lsp-mode
  (setq lsp-pyls-plugins-pycodestyle-enabled nil)
  (setq lsp-pyls-plugins-flake8-enabled t)

    ;; Add metals backend for lsp-mode
    (use-package lsp-metals
      :ensure t)
    (when (display-graphic-p)
      ;; Do any keybindings and theme setup here
      (use-package lsp-ui
        :ensure t
        :commands lsp-ui-mode)
      (use-package posframe
        :ensure t)
      (use-package dap-mode
        :ensure t
        :hook
        (lsp-mode . dap-mode)
        (lsp-mode . dap-ui-mode))
      (use-package helm-lsp
        :ensure t
        :config
        (define-key lsp-mode-map [remap xref-find-apropos] #'helm-lsp-workspace-symbol)
        )
      (use-package lsp-treemacs
        :ensure t)
      )
  (defun my/python-mode-setup ()
  (setq lsp-pylsp-configuration-sources ["pycodestyle"])
  (setq lsp-pylsp-plugins-pydocstyle-enabled t)
  (setq lsp-pylsp-plugins-pydocstyle-ignore ["D100" "D101" "D203", "D107", "D105", "D104", "D102", "D103", "D106", "D401", "D413", "D202", "D204", "D213", "D406", "D407", "D408", "D409", "D410", "D411", "D412", "D415", "D416", "D417", "D418", "D419", "D420", "D421", "D422", "D423", "D424", "D425", "D426", "D427", "D428", "D429", "D430", "D431", "D432", "D433", "D434", "D435", "D436", "D437", "D438", "D439", "D440", "D441", "D442", "D443", "D444", "D445", "D446", "D447", "D448", "D449", "D450", "D451", "D452", "D453", "D454", "D455", "D456", "D457", "D458", "D459", "D460", "D461", "D462", "D463", "D464", "D465", "D466", "D467", "D468", "D469", "D470", "D471", "D472", "D473", "D474", "D475", "D476", "D477", "D478", "D479", "D480", "D481", "D482", "D483", "D484", "D485", "D486", "D487", "D488", "D489", "D490", "D491", "D492", "D493", "D494", "D495", "D496", "D497", "D498", "D499", "D500", "D501", "D502", "D503", "D504", "D505", "D506", "D507", "D508", "D509", "D510", "D511", "D512", "D513", "D514", "D515", "D516", "D517", "D518", "D519", "D520", "D521", "D522"]))

(add-hook 'python-mode-hook #'my/python-mode-setup)
```


### 语法检测 {#语法检测}


#### flymake {#flymake}

```emacs-lisp
(remove-hook 'prog-mode-hook 'flymake-mode)
(remove-hook 'find-file-hook 'flymake-mode)
(use-package python
  :mode ("\\.py" . python-mode)
  :ensure t
  :config
  (flymake-mode) ;
)
;; ;; Redefine flymake-mode to do nothing
;; (defun flymake-mode (&optional arg)
;;   "Disable flymake mode."
;;   (interactive)
;;   ;; Do nothing
;;   )
```


#### flycheck {#flycheck}

```emacs-lisp
(use-package flycheck
  :ensure t
  :init
  (global-flycheck-mode)
  (setq flycheck-checkers '(lsp-ui))
  )
```


### 项目管理 {#项目管理}

```emacs-lisp
(use-package projectile
  :ensure t
  :config
  (projectile-global-mode)
  (define-key projectile-mode-map (kbd "C-c p") 'projectile-command-map)
  (setq projectile-switch-project-action 'projectile-dired)
  (setq projectile-project-search-path '("~/Gitlab/offline/" "~/Gitlab/online/" "~/Github/PrivateHub"))
  (projectile-register-project-type 'java '("pom.xml")
                                    :compile "mvn compile"
                                    :test "mvn test"
                                    :run "mvn package"
                                    :test-suffix "Test")
  )
```


### other {#other}

```emacs-lisp
(use-package quickrun
  :ensure t
  :config
  (quickrun-add-command "c++/c11"
    '((:command . "g++")
      (:exec    . ("%c -std=c++11 %o -o %e %s"
                   "%e %a"))
      (:remove  . ("%e")))
    :default "c++")
  )
;; Enable scala-mode and sbt-mode
;; eglot metal 的配置
;; (use-package eglot
;;   :pin melpa-stable
;;   ;; (optional) Automatically start metals for Scala files.
;;   ;; :hook (scala-mode . eglot-ensure)
;;   :config
;;   (setq eldoc-echo-area-use-multiline-p nil)
;;   ;; (add-hook 'eglot--managed-mode-hook (lambda () (flymake-mode -1)))
;;   )
;;需要安装metals-emacs
;; sudo coursier bootstrap \
;;   --java-opt -XX:+UseG1GC \
;;   --java-opt -XX:+UseStringDeduplication  \
;;   --java-opt -Xss4m \
;;   --java-opt -Xms100m \
;;   org.scalameta:metals_2.13:0.11.10 \
;;   -o /usr/local/bin/metals-emacs -f
(setq lsp-pyls-plugins-pycodestyle-enabled nil)
;; other important
(use-package wgrep
  :ensure t)
(use-package multiple-cursors
  :ensure t
  :bind (("C->"           . mc/mark-next-like-this)
         ("C-<"           . mc/mark-previous-like-this)
         ("C-M->"         . mc/skip-to-next-like-this)
         ("C-M-<"         . mc/skip-to-previous-like-this)
         ("C-c C-<"       . mc/mark-all-like-this)
         ("C-S-c C-S-c"   . mc/edit-lines)
         ("C-S-<mouse-1>" . mc/add-cursor-on-click)
         :map mc/keymap
         ("C-|" . mc/vertical-align-with-space))
  :config
  (setq mc/insert-numbers-default 1))
;; yasnippet
(when (display-graphic-p)
  (use-package auto-highlight-symbol
    :ensure t
    :config
    (global-auto-highlight-symbol-mode t)
    )
  (use-package projectile-rails
    :ensure t
    :config
    (projectile-rails-global-mode)
    )
  (use-package highlight-indent-guides
    :ensure t
    :config
    (add-hook 'python-mode-hook 'highlight-indent-guides-mode)
    (setq highlight-indent-guides-method 'character))
  )
```


### yasnippet {#yasnippet}

```emacs-lisp
  (use-package yasnippet
  :ensure t
  :config
  (setq yas-snippet-dirs
        '("~/Github/PrivateHub/tech_org/dotfiles/emacs_snippets"
          ))
  (yas-global-mode 1)
  )
(use-package yasnippet-snippets
  :ensure t
  )
```


### languages {#languages}


#### 各种小语言 {#各种小语言}

```emacs-lisp
(use-package protobuf-mode
  :ensure t
  )
(use-package yaml-mode
  :ensure t
  :config
  (add-to-list 'auto-mode-alist '("\\.yml\\'" . yaml-mode))
  )
(use-package go-mode
  :ensure t
  :config
  (autoload 'go-mode "go-mode" nil t)
  (add-to-list 'auto-mode-alist '("\\.go\\'" . go-mode))
  )
(use-package scala-mode
  :interpreter ("scala" . scala-mode))
;; Enable sbt mode for executing sbt commands
(use-package sbt-mode
  :commands sbt-start sbt-command
  :config
  ;; WORKAROUND: https://github.com/ensime/emacs-sbt-mode/issues/31
  ;; allows using SPACE when in the minibuffer
  (substitute-key-definition
   'minibuffer-complete-word
   'self-insert-command
   minibuffer-local-completion-map)
  ;; sbt-supershell kills sbt-mode:  https://github.com/hvesalai/emacs-sbt-mode/issues/152
  (setq sbt:program-options '("-Dsbt.supershell=false")))
```


#### python {#python}

```emacs-lisp
;; (when (display-graphic-p)
;;   (use-package elpy
;;     :ensure t
;;     :init
;;     (elpy-enable))
;;   )
(add-hook 'python-mode-hook
          (lambda ()
            (setq indent-tabs-mode nil)
            (setq tab-width 4)
            (setq python-indent-offset 4)))
(add-hook 'python-mode-hook
          (lambda ()
            ;; (setq flycheck-python-pylint-executable "/Users/mobvista/miniforge3/envs/tf26/bin/pylint")
            ;; (setq flycheck-pylintrc "~/.pylintrc")
            ;; (setq flycheck-flake8rc "~/.config/flake8")
            )
          )
;; 关于anaconda，先安装conda，使用conda-activate可以切换conda环境
(use-package conda
  :ensure t
  :init
  (setq conda-anaconda-home (expand-file-name "~/miniforge3/"))
  (setq conda-env-home-directory (expand-file-name "~/miniforge3"))
  (setq-default mode-line-format (cons mode-line-format '(:exec conda-env-current-name)))
  (if (display-graphic-p)
      (progn
      (conda-env-activate "py39")
   )
      (progn
      (conda-env-activate "py39")
   )
      )
  )
;; anaconda-mode定义了很多跳转功能，比如 anaconda-mode-find-definitions M.,M=
;; (use-package anaconda-mode
;;   :ensure t
;;   :bind (("C-c C-x" . next-error))
;;   :config
;;   (add-hook 'python-mode-hook 'anaconda-mode)
;;   (add-hook 'python-mode-hook 'anaconda-eldoc-mode)
;;   )
;; (use-package company-anaconda
;;   :ensure t
;;   :config
;;   (eval-after-load "company"
;;     '(add-to-list 'company-backends '(company-anaconda))))
;; (use-package company-shell
;;   :ensure t
;;   :config
;;   (eval-after-load "company"
;;     '(add-to-list 'company-backends '(company-shell company-shell-env))))
                                        ;(use-package company-jedi
                                        ;  :ensure t
                                        ;  )
                                        ;(use-package company-irony
                                        ;  :ensure t
                                        ;  :config
                                        ;  (eval-after-load "company"
                                        ;    '(add-to-list 'company-backends '(company-irony))))
;; PYTHON CONFIG END
;; 可以让imenu 平铺起来flat
(defun python-imenu-use-flat-index
    ()
  (setq imenu-create-index-function
        #'python-imenu-create-flat-index))
(add-hook 'python-mode-hook
          #'python-imenu-use-flat-index)
(use-package py-autopep8
  :hook ((python-mode) . py-autopep8-mode))
(use-package dap-mode
  :ensure t
  :config
  (dap-mode 1)
  ;; The modes below are optional
  (dap-ui-mode 1)
  ;; enables mouse hover support
  (dap-tooltip-mode 1)
  ;; use tooltips for mouse hover
  ;; if it is not enabled `dap-mode' will use the minibuffer.
  (tooltip-mode 1)
  ;; displays floating panel with debug buttons
  ;; requies emacs 26+
  (dap-ui-controls-mode 1)
  (require 'dap-python)
  ;; if you installed debugpy, you need to set this
  ;; https://github.com/emacs-lsp/dap-mode/issues/306
  (setq dap-python-debugger 'debugpy)
  )
(when (display-graphic-p)
  ;; Do any keybindings and theme setup here
  (use-package realgud
    :ensure t)
  (use-package ein
    :ensure t)
  (use-package python-pytest)
  )
;; run-python 使用ipython
(setq python-shell-completion-native-enable nil)
(when (executable-find "ipython")
  (setq python-shell-interpreter "ipython"))
```


#### c++ {#c-plus-plus}

```emacs-lisp
(require 'cc-mode)
(add-hook 'c++-mode-hook
          (lambda () (setq flycheck-clang-language-standard "c++11")))
(use-package smartparens
  :ensure t
  :config
  (add-hook 'prog-mode-hook 'smartparens-mode)
  )
(use-package smartparens-config
  :ensure smartparens
  :config (progn (show-smartparens-global-mode t)))
(setq smartparens-strict-mode nil)
(defmacro def-pairs (pairs)
  "Define functions for pairing. PAIRS is an alist of (NAME . STRING)
conses, where NAME is the function name that will be created and
STRING is a single-character string that marks the opening character.
  (def-pairs ((paren . \"(\")
              (bracket . \"[\"))
defines the functions WRAP-WITH-PAREN and WRAP-WITH-BRACKET,
respectively."
  `(progn
     ,@(loop for (key . val) in pairs
             collect
             `(defun ,(read (concat
                             "wrap-with-"
                             (prin1-to-string key)
                             "s"))
                  (&optional arg)
                (interactive "p")
                (sp-wrap-with-pair ,val)))))
(def-pairs ((paren . "(")
            (bracket . "[")
            (brace . "{")
            (single-quote . "'")
            (double-quote . "\"")
            (back-quote . "`")))
(bind-keys
 :map smartparens-mode-map
 ;; ("C-M-a" . sp-beginning-of-sexp)
 ;; ("C-M-e" . sp-end-of-sexp)
 ("C-<down>" . sp-down-sexp)
 ("C-<up>"   . sp-up-sexp)
 ("M-<down>" . sp-backward-down-sexp)
 ("M-<up>"   . sp-backward-up-sexp)
 ("C-M-f" . sp-forward-sexp)
 ("C-M-b" . sp-backward-sexp)
 ("C-M-n" . sp-next-sexp)
 ("C-M-p" . sp-previous-sexp)
 ("C-S-f" . sp-forward-symbol)
 ("C-S-b" . sp-backward-symbol)
 ("C-<right>" . sp-forward-slurp-sexp)
 ("M-<right>" . sp-forward-barf-sexp)
 ("C-<left>"  . sp-backward-slurp-sexp)
 ("M-<left>"  . sp-backward-barf-sexp)
 ("C-M-t" . sp-transpose-sexp)
 ("C-M-k" . sp-kill-sexp)
 ("C-k"   . sp-kill-hybrid-sexp)
 ("M-k"   . sp-backward-kill-sexp)
 ("C-M-w" . sp-copy-sexp)
 ("C-M-d" . sp-delete-sexp)
 ("M-<backspace>" . backward-kill-word)
 ("C-<backspace>" . sp-backward-kill-word)
 ([remap sp-backward-kill-word] . backward-kill-word)
 ("M-[" . sp-backward-unwrap-sexp)
 ("M-]" . sp-unwrap-sexp)
 ("C-x C-t" . sp-transpose-hybrid-sexp)
 ("s-i w("  . wrap-with-parens)
 ("s-i w["  . wrap-with-brackets)
 ("s-i w{"  . wrap-with-braces)
 ("s-i w'"  . wrap-with-single-quotes)
 ("s-i w\"" . wrap-with-double-quotes)
 ("s-i w_"  . wrap-with-underscores)
 ("s-i w`"  . wrap-with-back-quotes))
(defun sp-pair-curly-down-sexp (&optional arg)
  (interactive "P")
  (sp-restrict-to-pairs "{" 'sp-down-sexp))
(defun sp-pair-curly-up-sexp (&optional arg)
  (interactive "P")
  (sp-restrict-to-pairs "}" 'sp-up-sexp))
(define-key c++-mode-map (kbd "s-i }") (sp-restrict-to-pairs-interactive "{" 'sp-down-sexp))
(define-key c++-mode-map (kbd "s-i {") (sp-restrict-to-pairs-interactive "}" 'sp-up-sexp))
(define-key c++-mode-map (kbd "s-i u") (sp-restrict-to-pairs-interactive "{" 'sp-backward-up-sexp))
;; for cmakelist.txt
(when (display-graphic-p)
  ;; Do any keybindings and theme setup here
  (use-package cmake-mode
    :ensure t
    :config
    (setq auto-mode-alist
          (append '(("CMakeLists\\.txt\\'" . cmake-mode)
                    ("\\.cmake\\'" . cmake-mode))
                  auto-mode-alist))
    )
  )
;; (use-package cmake-idle
;;   :ensure t
;;   :config
;;   (cmake-ide-setup)
;;   )
;; c++
;; (helm-mode)
;; (require 'helm-xref)
;; (define-key global-map [remap find-file] #'helm-find-files)
;; (define-key global-map [remap execute-extended-command] #'helm-M-x)
;; (define-key global-map [remap switch-to-buffer] #'helm-mini)
;; (which-key-mode)
;; (add-hook 'c-mode-hook 'lsp)
;; (add-hook 'c++-mode-hook 'lsp)
;; (setq gc-cons-threshold (* 100 1024 1024)
;;       read-process-output-max (* 1024 1024)
;;       treemacs-space-between-root-nodes nil
;;       company-idle-delay 0.0
;;       company-minimum-prefix-length 1
;;       lsp-idle-delay 0.1)  ;; clangd is fast
;; (with-eval-after-load 'lsp-mode
;;   (add-hook 'lsp-mode-hook #'lsp-enable-which-key-integration)
;;   (require 'dap-cpptools)
;;   (yas-global-mode))
;; (require 'lsp-docker)
;; (defvar lsp-docker-client-packages
;;     '(lsp-css lsp-clients lsp-bash lsp-go lsp-pylsp lsp-html lsp-typescript
;;       lsp-terraform lsp-clangd))
;; (setq lsp-docker-client-configs
;;     '((:server-id bash-ls :docker-server-id bashls-docker :server-command "bash-language-server start")
;;       (:server-id clangd :docker-server-id clangd-docker :server-command "clangd")
;;       (:server-id css-ls :docker-server-id cssls-docker :server-command "css-languageserver --stdio")
;;       (:server-id dockerfile-ls :docker-server-id dockerfilels-docker :server-command "docker-langserver --stdio")
;;       (:server-id gopls :docker-server-id gopls-docker :server-command "gopls")
;;       (:server-id html-ls :docker-server-id htmls-docker :server-command "html-languageserver --stdio")
;;       (:server-id pylsp :docker-server-id pyls-docker :server-command "pylsp")
;;       (:server-id ts-ls :docker-server-id tsls-docker :server-command "typescript-language-server --stdio")))
;; (lsp-docker-init-clients
;;   :path-mappings '(("~/Gitlab/offline/ltv_model" . "/projects"))
;;   :client-packages lsp-docker-client-packages
;;   :client-configs lsp-docker-client-configs)
```


## org {#org}


### appearance {#appearance}


#### org basic {#org-basic}

```emacs-lisp
(defun dw/org-mode-setup ()
  (org-indent-mode)
  (variable-pitch-mode 1)
  (auto-fill-mode 0)
  (visual-line-mode 1)
  (setq evil-auto-indent nil)
  (diminish org-indent-mode))
(use-package org
  :defer t
  :hook (org-mode . dw/org-mode-setup)
  :config
  (setq org-ellipsis " ▾"
        org-hide-emphasis-markers t
        org-src-fontify-natively t
        org-fontify-quote-and-verse-blocks t
        org-src-tab-acts-natively t
        org-edit-src-content-indentation 2
        org-hide-block-startup nil
        org-src-preserve-indentation nil
        org-startup-folded 'content
        org-cycle-separator-lines 2
        org-capture-bookmark nil)
  (setq inferior-lisp-program "/opt/homebrew/bin/sbcl")
  (setq org-imenu-depth 4)
  (org-babel-do-load-languages
   'org-babel-load-languages
   '((emacs-lisp . t)
     (python . t)
     ))
  (push '("conf-unix" . conf-unix) org-src-lang-modes)
  (add-to-list 'org-file-apps
               '("\\.pdf\\'" . (lambda (file link)
                                 (find-file file))))
  )
```


#### table 中文 {#table-中文}

```emacs-lisp
;; (use-package valign
;;   :hook (org-mode . valign-mode)
;;   :config
;;   (setq valign-fancy-bar t)
;;   )
```


#### 页面宽度 (visual-fill-column) {#页面宽度--visual-fill-column}

```emacs-lisp
(defun my/org-mode-visual-fill ()
(interactive)
  (setq visual-fill-column-width 150
        visual-fill-column-center-text t)
  (visual-fill-column-mode 1))
(use-package visual-fill-column
  :hook (org-mode . my/org-mode-visual-fill))
```


#### 插入图片 {#插入图片}

```emacs-lisp
(use-package org-download
  :bind ("C-S-y" . org-download-clipboard)
  :config
  (add-hook 'dired-mode-hook 'org-download-enable)
  (setq-default org-download-heading-lvl nil)
  ;; (setq org-image-actual-width 600)
  (setq-default org-download-image-dir "./images"))
```


#### 字体 {#字体}

```emacs-lisp
(setq-default fill-column 80)
(set-fontset-font t 'symbol "Apple Color Emoji" nil 'prepend)
(setq org-image-actual-width nil)
(setq org-html-htmlize-output-type nil)
(add-hook 'org-mode-hook
          (lambda ()
            (set-face-attribute 'default nil :font "Menlo-15")))
(require 'org-faces)
(set-face-attribute 'org-document-title nil :font "Menlo-15" :weight 'bold :height 2.7)
(dolist (face '((org-level-1 . 1.2)
                (org-level-2 . 1.06)
                (org-level-3 . 1.03)
                (org-level-4 . 1.0)
                (org-level-5 . 1.0)
                (org-level-6 . 1.0)
                (org-level-7 . 1.0)
                (org-level-8 . 1.0)
                (org-table . 1.0)
                ))
  (set-face-attribute (car face) nil :font "Menlo-15" :weight 'medium :height (cdr face)))
(require 'org-indent)
(set-face-attribute 'fixed-pitch nil
                    :font "Menlo-15"
                    :weight 'light
                    :height 1.0)
(set-face-attribute 'variable-pitch nil
                    ;; :font "Cantarell"
                    :font "Menlo-15"
                    :height 1.0
                    :weight 'light)
(set-face-attribute 'org-block nil :foreground nil :inherit 'fixed-pitch)
(set-face-attribute 'org-table nil  :inherit 'fixed-pitch)
;; (set-face-attribute 'org-table nil :height 0.95)
(set-face-attribute 'org-formula nil  :inherit 'fixed-pitch)
(set-face-attribute 'org-code nil   :inherit '(shadow fixed-pitch))
(set-face-attribute 'org-indent nil :inherit '(org-hide fixed-pitch))
(set-face-attribute 'org-verbatim nil :inherit '(shadow fixed-pitch))
(set-face-attribute 'org-special-keyword nil :inherit '(font-lock-comment-face fixed-pitch))
(set-face-attribute 'org-meta-line nil :inherit '(font-lock-comment-face fixed-pitch))
(set-face-attribute 'org-checkbox nil :inherit 'fixed-pitch)
;; Get rid of the background on column views
(set-face-attribute 'org-column nil :background nil)
(set-face-attribute 'org-column-title nil :background nil)
(require 'color)
(set-face-attribute 'org-block nil :background
                    (color-lighten-name
                     (face-attribute 'default :background) 25))
;; (custom-set-faces
;;  '(org-block-begin-line
;;    ((t (:background "lightgreen" :foreground "black"))))
;;  '(org-block-end-line
;;    ((t (:background "lightgreen" :foreground "black")))))
(set-face-attribute 'org-block-begin-line  nil :background
                    (color-lighten-name
                     (face-attribute 'default :background) 50))
(set-face-attribute 'org-block-end-line  nil :background
                    (color-lighten-name
                     (face-attribute 'default :background) 50))
```


#### prettify-symbols-alist {#prettify-symbols-alist}

```emacs-lisp
(add-hook 'org-mode-hook  (lambda ()
                            (setq prettify-symbols-alist
                                  '(
                                    ;; ("lambda" . ?λ)
                                    ("#+TITLE:" . "📗")
                                    ("#+AUTHOR:" . "👤")
                                    ("#+DATE:" . "📅")
                                    ("#+EMAIL:" . "🌐")
                                    ("TODO" . "📌")
                                    ("DONE" . "✅")
                                    ("1) " . "1️⃣")
                                    ("2) " . "2️⃣")
                                    ("3) " . "3️⃣")
                                    ("4) " . "4️⃣")
                                    ("#+begin_example" . "📎")
                                    ("#+end_example" . "📎")
                                    ("#+BEGIN_EXAMPLE" . "📎")
                                    ("#+END_EXAMPLE" . "📎")
                                    ("#+begin_quote" . "️🖋️️")
                                    ("#+end_quote" . "🖋️️")
                                    ("#+BEGIN_QUOTE" . "🖋️️")
                                    ("#+END_QUOTE" . "🖋️️")
                                    ("#+begin_src" . "🖥️")
                                    ;; ("#+end_src" . "🛠️")
                                    ("#+end_src" . "🖥️")
                                    ("#+BEGIN_SRC" . "🖥️")
                                    ("#+END_SRC" . "🖥️")
                                    ("#+RESULTS:" . "💎")
                                    ))
                            (prettify-symbols-mode)
                            )
          )
```


#### org-superstar {#org-superstar}

```emacs-lisp
(use-package org-superstar
  :ensure t
  :after org
  :hook (org-mode . org-superstar-mode)
  :custom
  (org-superstar-remove-leading-stars t)
  ;; (org-superstar-headline-bullets-list '("◉" "○" "●" "○" "●" "○" "●"))
  :config
  (setq org-superstar-headline-bullets-list '("⭐" "💫" "✨" "🌟"  "🌞" "🔯" "✴️" ))
  (setq org-superstar-item-bullet-alist '((?- . "🔵")
                                          (?+ . "🔘")
                                          (?* . "⚪")
                                          ))
  )
```


#### html {#html}

```emacs-lisp
(use-package htmlize
  :ensure t)
(setq org-html-htmlize-output-type 'css)
(setq org-html-htmlize-output-type 'inline-css)
(require 'ox-html)
(setq org-html-htmlize-output-type 'css)
(setq org-hierarchical-todo-statistics nil)
(require 'epa-file)
(epa-file-enable)
(setq org-format-latex-options (plist-put org-format-latex-options :scale 1.8))
```


### presentation {#presentation}


#### org-tree-slide <span class="tag"><span class="tree">tree</span><span class="slide">slide</span></span> {#org-tree-slide}

```emacs-lisp
(use-package hide-mode-line)
(defun efs/presentation-setup ()
  (setq org-format-latex-options (plist-put org-format-latex-options :scale 2.1))
  (org-latex-preview '(16))
  (setq visual-fill-column-width 110
        visual-fill-column-center-text t) ; 调整显示界面
  ;; Center the presentation and wrap lines
  (visual-fill-column-mode 1)
  (visual-line-mode 1)
  (setq header-line-format " ") ; 在标题前加入空行
  (display-line-numbers-mode 0)
  (org-display-inline-images)
  (tooltip-mode -1)			;可以在鼠标hover到图片的时候不显示图片的名字
  ;; Scale the text.  The next line is for basic scaling:
  ;; (setq text-scale-mode-amount 3)
  (setq-local face-remapping-alist '((default (:height 1.3 ) variable-pitch)
                                     (header-line (:height 1.8 ) variable-pitch)
                                     (org-document-title (:height 2.0) org-document-title)
                                     (org-level-1 (:height 1.5 ) org-level-1)
                                     (org-level-2 (:height 1.35 )  org-level-2)
                                     (org-level-3 (:height 1.05 )  org-level-3)
                                     (org-code (:height 1.0 ) org-code)
                                     (org-verbatim (:height 0.9 ) org-verbatim)
                                     (org-block (:height 0.8 ) org-block)
                                     (org-block-begin-line (:height 1.05) org-block)
                                     (org-list-dt (:height 1.2) org-list-dt)
                                     )
              )
  )
(defun efs/presentation-end ()
  ;; (hide-mode-line-mode 0)
  (setq-local face-remapping-alist '((default variable-pitch default)))
  (setq org-format-latex-options (plist-put org-format-latex-options :scale 1.8))
  (tooltip-mode 1)
  )
(defun my-make-invisible (my-re)
  (interactive "sRE Search Term: ")
  (save-excursion
    (goto-char (point-min))
    (while (re-search-forward my-re nil t)
      (setq invisible-overlay (make-overlay (line-beginning-position) (line-end-position)))
      (overlay-put invisible-overlay 'invisible t))))
(defun my-make-visible (my-re)
  (interactive "sRE Search Term: ")
  (save-excursion
    (goto-char (point-min))
    (while (re-search-forward my-re nil t)
      (let ((overlays (overlays-in (line-beginning-position) (line-end-position))))
        (dolist (overlay overlays)
          (when (overlay-get overlay 'invisible)
            (delete-overlay overlay)))))))
(defun my-org-tree-slide-hide-lines ()
  (my-make-invisible "#\\+ATTR_HTML\\|#\\+DOWNLOADED")
  )
(defun my-org-tree-slide-show-lines ()
  (my-make-visible "#\\+ATTR_HTML\\|#\\+DOWNLOADED")
  )
(use-package org-tree-slide
  :hook ((org-tree-slide-play . efs/presentation-setup)
         (org-tree-slide-stop . efs/presentation-end))
  :custom
  (org-tree-slide-slide-in-effect nil)
  (org-tree-slide-activate-message "Presentation started!")
  (org-tree-slide-deactivate-message "Presentation finished!")
  (org-tree-slide-header nil)
  (org-tree-slide-breadcrumbs " > ")
  (org-tree-slide-skip-outline-level 3)	;; 只对level1,2做slides
  (org-tree-slide-heading-emphasis t)
  (org-tree-slide-fold-subtrees-skipped	t) ;; 是否对level3 以下的subtree做折叠
  (org-tree-slide-cursor-init t)
  (org-image-actual-width nil)
  :config
  (add-hook 'org-tree-slide-play-hook 'my-org-tree-slide-hide-lines)
  (add-hook 'org-tree-slide-stop-hook 'my-org-tree-slide-show-lines)
  (global-set-key (kbd "<f8>") 'org-tree-slide-mode)
  :bind (([f8] . org-tree-slide-mode)
         )
  )
(with-eval-after-load 'org-tree-slide
  (define-key org-tree-slide-mode-map (kbd "<left>") 'org-tree-slide-move-previous-tree)
  (define-key org-tree-slide-mode-map (kbd "<right>") 'org-tree-slide-move-next-tree)
  (define-key org-tree-slide-mode-map (kbd "<up>") 'org-previous-visible-heading)
  (define-key org-tree-slide-mode-map (kbd "<down>") 'org-next-visible-heading)
  (define-key org-tree-slide-mode-map (kbd "C->") 'mc/mark-next-like-this)
  (define-key org-tree-slide-mode-map (kbd "C-<") 'mc/mark-previous-like-this)
  )
```


#### org-present {#org-present}

```emacs-lisp
(use-package org-present
  :config
  (defun my/org-present-prepare-slide (buffer-name heading)
    (org-overview)  ; 仅显示顶层标题Show only top-level headlines
    (org-show-entry); 展开当前标题Unfold the current entry
    (org-show-children))   ; 显示当前子标题
  (defun my/org-present-start () ; 开始幻灯片的设置
    ;; (turn-off-evil-mode)
    (org-present-hide-cursor)		; 隐藏光标
    (org-latex-preview '(16))
    (setq visual-fill-column-width 110
          visual-fill-column-center-text t) ; 调整显示界面
    ;; Center the presentation and wrap lines
    (visual-fill-column-mode 1)
    (visual-line-mode 1)
    ;; 调整字体大小
    (setq-local face-remapping-alist '((default (:height 1.1) variable-pitch)
                                       (header-line (:height 3.0) variable-pitch)
                                       (org-document-title (:height 3.0) org-document-title)
                                       (org-level-1 (:height 2.0) org-level-1)
                                       (org-level-2 (:height 1.5) org-level-2)
                                       (org-code (:height 0.8 ) org-code)
                                       (org-verbatim (:height 0.95) org-verbatim)
                                       (org-block (:height 0.8 ) org-block)
                                       (org-block-begin-line (:height 0.7) org-block)))
    (setq header-line-format " ") ; 在标题前加入空行
    (display-line-numbers-mode 0)
    (org-display-inline-images) ; 显示图片
    (read-only-mode 1)
    (org-show-all)
    ) ; 只读模式
  (defun my/org-present-end () ; 重置上述设置
    (setq-local face-remapping-alist
                '((default variable-pitch default)))
    (org-present-show-cursor)
    (org-latex-preview '(64))
    (setq header-line-format nil)
    (org-remove-inline-images)
    (org-present-small)
    (read-only-mode 0)
    (display-line-numbers-mode 1)
    )
  (defun my/org-present-hide-tagged-entries ()
    (interactive)
    (save-excursion
      (goto-char (point-min))
      (while (re-search-forward ":unused:" nil t)
        (org-hide-entry))))
  (add-hook 'org-present-mode-hook 'my/org-present-start)
  (add-hook 'org-present-mode-hook
            (lambda ()
              (my/org-present-hide-tagged-entries)))
  (add-hook 'org-present-mode-quit-hook 'my/org-present-end)
  ;; (add-hook 'org-present-after-navigate-functions 'my/org-present-prepare-slide)
  )
```


### org-capture {#org-capture}

```emacs-lisp
;; about org-capture
(setq my-todo-file "~/Documents/OrgDoc/todos.org")
(setq my-idea-file "~/Documents/OrgDoc/ideas.org")
(setq org-agenda-files (list
                        my-todo-file
                        "~/Github/PrivateHub/tech_org/Linux/Emacs.org"
                        "~/Github/PrivateHub/tech_org/Linux/mac.org"
                        ))
(require 'helm)
(helm-mode 1)
(setq org-refile-targets '((nil :maxlevel . 1)
                           (org-agenda-files :maxlevel . 1)))
(setq org-completion-use-ido nil)
(setq org-outline-path-complete-in-steps nil)
(setq org-refile-use-outline-path 'file)
(define-key global-map "\C-ca" 'org-agenda)
(define-key global-map "\C-cc" 'org-capture)
(setq org-default-notes-file (concat org-directory "~/notes.org"))
(setq org-capture-templates nil)
(add-to-list 'org-capture-templates '("t" "TODOS"))
(add-to-list 'org-capture-templates
             '("th" "TODO HOME" entry
               (file+headline my-todo-file "Home")
               "* TODO %^{任务名}\n%u\n" ))
(add-to-list 'org-capture-templates
             '("tt" "TODO TECH" entry
               (file+headline my-todo-file "Technology")
               "* TODO %^{任务名}\n%u\n"))
(add-to-list 'org-capture-templates
             '("te" "TODO ECONOMY" entry
               (file+headline my-todo-file "ECONOMY")
               "* TODO %^{任务名}\n%u\n"))
(add-to-list 'org-capture-templates '("i" "IDEAS"))
(add-to-list 'org-capture-templates
             '("it" "关于技术的思考" entry
               (file+headline my-idea-file "Technology")
               "* %^{heading}\n%U\n"))
(add-to-list 'org-capture-templates
             '("il" "关于人生的思考" entry
               (file+headline my-idea-file "Life")
               "* %^{heading}\n%U\n"))
(add-to-list 'org-capture-templates
             '("j" "Journal" entry (file+datetree "~/Documents/OrgDoc/journal.org")
               "* %U - %^{heading}\n  %?"))
(add-to-list 'org-capture-templates
             '("s" "Skills" entry (file "~/Documents/OrgDoc/skill.org")
               "* %^{heading} %t %^g\n  %?\n"))
(add-to-list 'org-capture-templates
             '("b" "Billing" plain
               (file+function "~/Documents/OrgDoc/billing.org" find-month-tree)
               " | %U | %^{类别} | %^{描述} | %^{金额} |" :kill-buffer t))
(defun get-year-and-month ()
  (list (format-time-string "%Y年") (format-time-string "%m月")))
(defun find-month-tree ()
  (let* ((path (get-year-and-month))
         (level 1)
         end)
    (unless (derived-mode-p 'org-mode)
      (error "Target buffer \"%s\" should be in Org mode" (current-buffer)))
    (goto-char (point-min))             ;移动到 buffer 的开始位置
    ;; 先定位表示年份的 headline，再定位表示月份的 headline
    (dolist (heading path)
      (let ((re (format org-complex-heading-regexp-format
                        (regexp-quote heading)))
            (cnt 0))
        (if (re-search-forward re end t)
            (goto-char (point-at-bol))  ;如果找到了 headline 就移动到对应的位置
          (progn                        ;否则就新建一个 headline
            (or (bolp) (insert "\n"))
            (if (/= (point) (point-min)) (org-end-of-subtree t t))
            (insert (make-string level ?*) " " heading "\n"))))
      (setq level (1+ level))
      (setq end (save-excursion (org-end-of-subtree t t))))
    (org-end-of-subtree)))
```


### edit {#edit}

```emacs-lisp
(require 'org-tempo)
(add-to-list 'org-structure-template-alist '("sh" . "src sh"))
(add-to-list 'org-structure-template-alist '("el" . "src emacs-lisp"))
(add-to-list 'org-structure-template-alist '("li" . "src lisp"))
(add-to-list 'org-structure-template-alist '("sc" . "src scheme"))
(add-to-list 'org-structure-template-alist '("ts" . "src typescript"))
(add-to-list 'org-structure-template-alist '("py" . "src python"))
(add-to-list 'org-structure-template-alist '("go" . "src go"))
(add-to-list 'org-structure-template-alist '("yaml" . "src yaml"))
(add-to-list 'org-structure-template-alist '("json" . "src json"))
(add-to-list 'org-structure-template-alist '("chat" . "ai"))
(add-to-list 'org-structure-template-alist '("tangle" . "src emacs-lisp :tangle ~/Github/PrivateHub/linux_confs/emacs_personal/org_conf.el  :comments link"))
```


### my config {#my-config}

```emacs-lisp
(defun todo-to-int (todo)
  (cl-first (-non-nil
             (mapcar (lambda (keywords)
                       (let ((todo-seq
                              (-map (lambda (x) (cl-first (split-string  x "(")))
                                    (cl-rest keywords))))
                         (cl-position-if (lambda (x) (string= x todo)) todo-seq)))
                     org-todo-keywords))))
(defun my/org-sort-key ()
  (let* ((todo-max (apply #'max (mapcar #'length org-todo-keywords)))
         (todo (org-entry-get (point) "TODO"))
         (todo-int (if todo (todo-to-int todo) todo-max))
         (priority (org-entry-get (point) "PRIORITY"))
         (priority-int (if priority (string-to-char priority) org-default-priority)))
    (format "%03d %03d" todo-int priority-int)
    ))
(defun my/org-sort-entries ()
  (interactive)
  (org-sort-entries nil ?f #'my/org-sort-key))
  (defun my-tbl-export (name)
    "Search for table named `NAME` and export."
    (interactive "s")
    (outline-show-all)
    (let ((case-fold-search t))
      (if (search-forward-regexp (concat "#\\+NAME: +" name) nil t)
          (progn
            (next-line)
            (org-table-export (format "%s.csv" name) "orgtbl-to-csv")))))
(defun org-babel-goto-src-block-end ()
  "Move the point to the end of the current Org-mode source block."
  (interactive)
  ;; Go to the beginning of the source block
  (org-babel-goto-src-block-head)
  ;; Enable case-insensitive searching
  (let ((case-fold-search t))
    ;; Move forward to the end of the block
    (while (and (not (looking-at "#\\+end_src")) (not (eobp)))
      (forward-line 1))
    (when (looking-at "#\\+end_src")
      ;; Move to the end of the #+END_SRC line
      (end-of-line))))
```


### denote {#denote}

```emacs-lisp
(use-package denote
  :bind
  (("C-c n n" . denote)
   ("C-c n i" . denote-link-or-create)
   ("C-c n I" . denote-link)
   ("C-c n b" . denote-link-backlinks)
   ("C-c n a" . denote-add-front-matter)
   ("C-c n r" . denote-rename-file)
   ("C-c n R" . denote-rename-file-using-front-matter)
   )
  )
(setq denote-directory (expand-file-name "~/Github/PrivateHub/tech_org/denotes/")
      denote-known-keywords '("emacs" "python" "llm" "history")
      denote-infer-keywords t
      denote-sort-keywords t
      denote-allow-multi-word-keywords t
      denote-date-prompt-use-org-read-date t
      denote-link-fontify-backlinks t
      denote-front-matter-date-format 'org-timestamp
      denote-prompts '(title keywords))
;; 在work目录下创建标签为work的笔记
(defun my-work-notes ()
  "Create an entry tagged 'journal', while prompting for a title."
  (interactive)
  (denote
   (denote--title-prompt)
   '("work") 'denote-file-type '"./work"))
```


### helm-org {#helm-org}

这个可以方便快速的在headline中查看org的内容

```emacs-lisp
(use-package helm-org
  :ensure t
  :after org
  :bind (("C-c ho" . helm-org-in-buffer-headings))
  :custom
  (add-to-list 'helm-completing-read-handlers-alist '(org-capture . helm-org-completing-read-tags))
  (add-to-list 'helm-completing-read-handlers-alist '(org-set-tags . helm-org-completing-read-tags))
  )
```


### zotero {#zotero}

```emacs-lisp
(use-package zotxt
:ensure t
:init
(add-hook 'org-mode-hook #'org-zotxt-mode)
)
```


### agenda {#agenda}

```emacs-lisp
(use-package org-super-agenda
  :config
  (setq org-agenda-custom-commands
      '(("z" "Hugo view"
         ((agenda "" ((org-agenda-span 'day)
                      (org-super-agenda-groups
                       '((:name "Today"
                          :time-grid t
                          :date today
                          :todo "TODAY"
                          :scheduled today
                          :order 1)))))
          (alltodo "" ((org-agenda-overriding-header "")
                       (org-super-agenda-groups
                        '(;; Each group has an implicit boolean OR operator between its selectors.
                          (:name "Today"
                           :deadline today
                           :face (:background "black"))
                          (:name "Passed deadline"
                           :and (:deadline past :todo ("TODO" "WAITING" "HOLD" "NEXT"))
                           :face (:background "#7f1b19"))
                          (:name "Work important"
                           :and (:priority>= "B" :category "Work" :todo ("TODO" "NEXT")))
                          (:name "Work other"
                           :and (:category "Work" :todo ("TODO" "NEXT")))
                          (:name "Important"
                           :priority "A")
                          (:priority<= "B"
                           ;; Show this section after "Today" and "Important", because
                           ;; their order is unspecified, defaulting to 0. Sections
                           ;; are displayed lowest-number-first.
                           :order 1)
                          (:name "Papers"
                           :file-path "org/roam/notes")
                          (:name "Waiting"
                           :todo "WAITING"
                           :order 9)
                          (:name "On hold"
                           :todo "HOLD"
                           :order 10)))))))))
  (add-hook 'org-agenda-mode-hook 'org-super-agenda-mode)

  )
```


### org-babel {#org-babel}

```emacs-lisp
(org-babel-do-load-languages
 'org-babel-load-languages
 '((python . t)))
(setq inferior-lisp-program "/opt/homebrew/bin/sbcl")
```


### auctex {#auctex}

```emacs-lisp
(use-package tex
  :ensure auctex
  :config
  (setq TeX-auto-save t)
  (setq TeX-parse-self t)
  (setq TeX-save-query nil)
  (setq TeX-PDF-mode t)
  (setq TeX-source-correlate-mode t)
  (setq TeX-source-correlate-method 'synctex)
  (setq TeX-view-program-selection '((output-pdf "PDF Tools")))
  (setq TeX-view-program-list
        '(("PDF Tools" TeX-pdf-tools-sync-view))))
```


### cdlatex {#cdlatex}

```emacs-lisp
(use-package cdlatex
  :ensure t
  :config
  (add-hook 'LaTeX-mode-hook 'turn-on-cdlatex)
  (add-hook 'org-mode-hook #'org-cdlatex-mode)
  (setq org-format-latex-options (plist-put org-format-latex-options :scale 1.8))
  (setq cdlatex-math-modify-alist
        '((?D "\\mathbb" nil nil nil nil))))
```


### encrypt {#encrypt}

```emacs-lisp
;; brew install gpg
(require 'org-crypt)
;;当被加密的部份要存入硬碟时，自动加密回去
(org-crypt-use-before-save-magic)
;;设定要加密的tag 标签为secret
(setq org-crypt-tag-matcher "secret")
;;避免secret 这个tag 被子项目继承造成重复加密
;; (setq org-tags-exclude-from-inheritance (quote  ("secret")))
;;用于加密的GPG 金钥
;;可以设定任何ID 或是设成nil 来使用对称式加密(symmetric encryption)
(setq org-crypt-key nil)
(defun ag/reveal-and-move-back ()
  (org-reveal)
  (goto-char ag/old-point))
(defun ag/org-reveal-after-save-on ()
  (setq ag/old-point (point))
  (add-hook 'after-save-hook 'ag/reveal-and-move-back))
(defun ag/org-reveal-after-save-off ()
  (remove-hook 'after-save-hook 'ag/reveal-and-move-back))
(add-hook 'org-babel-pre-tangle-hook 'ag/org-reveal-after-save-on)
(add-hook 'org-babel-post-tangle-hook 'ag/org-reveal-after-save-off)
```


### my funs {#my-funs}


#### my/org-kill-list-item {#my-org-kill-list-item}

```emacs-lisp
(defun my/org-kill-list-item (&optional delete)
  "Kill list item at POINT.
Delete if DELETE is non-nil.
In interactive calls DELETE is the prefix arg."
  (interactive "P")
  (unless (org-at-item-p) (error "Not at an item"))
  (let* ((col (current-column))
         (item (point-at-bol))
         (struct (org-list-struct)))
    (org-list-send-item item (if delete 'delete 'kill) struct)
    (org-list-write-struct struct (org-list-parents-alist struct))
    (org-list-repair)
    (org-move-to-column col)))

```


#### my/org-which {#my-org-which}

```emacs-lisp
(defun my/org-which-function ()
  (interactive)
  (when (eq major-mode 'org-mode)
    (concat (mapconcat 'identity (org-get-outline-path t) " > ") "         ")
    ))
;; this is for doom modeline configuration
(setq global-mode-string (list '(:eval (my/org-which-function))))
```


## keybiddings {#keybiddings}

```emacs-lisp
(global-set-key "\C-xf" 'helm-recentf)
(global-set-key "\C-xg" 'magit-status)
(global-set-key "\C-z" 'set-mark-command)
(global-set-key (kbd "C-x m") 'eshell)
(global-set-key (kbd "C-c hs") 'helm-swoop)
(use-package general
  :ensure t
  :config
  (general-create-definer my-leader-def
    :prefix "s-i")
  (general-create-definer my-ctrl-leader-def
    :prefix "C-c")
  (my-leader-def
    "is" 'swiper-isearch
    "*" 'isearch-forward-symbol-at-point
    "ttl" 'toggle-truncate-lines
    "sw" 'ace-swap-window
    "tf" 'transpose-frame
    "rf" 'rotate-frame
    "fi" 'crux-find-user-init-file
    "hs" 'hs-show-block
    "hh" 'hs-hide-block
    "oti" 'org-toggle-item
    "oth" 'org-toggle-heading
    "os" 'org-tree-slide-mode
    "ots" 'org-tree-slide-mode
    "C-f" 'outline-forward-same-level
    "C-b" 'outline-backward-same-level
    "C-n" 'outline-next-heading
    "C-p" 'outline-previouse-heading
    "C-u" 'outline-up-heading
    ))
(use-package move-text
  :ensure t
  :config
  (my-ctrl-leader-def
    "<up>" 'move-text-up
    "<down>" 'move-text-down)
  (global-set-key [s-up] 'move-text-up)
  (global-set-key [s-down] 'move-text-down)
  )
(use-package crux
  :ensure t
  :config
  (global-set-key [(shift return)] 'crux-smart-open-line)
  (global-set-key (kbd "M-o") 'crux-smart-open-line)
  (global-set-key [(control shift return)] 'crux-smart-open-line-above)
  ;; (global-set-key (kbd "C-c n") 'crux-cleanup-buffer-or-region)
  (global-set-key (kbd "C-M-z") 'crux-indent-defun)
  (global-set-key (kbd "C-c u") 'crux-view-url)
  (global-set-key (kbd "C-c D") 'crux-delete-file-and-buffer)
  (global-set-key (kbd "C-c d") 'crux-duplicate-current-line-or-region)
  (global-set-key (kbd "C-c M-d") 'crux-duplicate-and-comment-current-line-or-region)
  (global-set-key (kbd "C-c r") 'crux-rename-buffer-and-file)
  (global-set-key (kbd "C-c t") 'vterm)
  (global-set-key (kbd "C-c k") 'crux-kill-other-buffers)
  (global-set-key (kbd "C-c TAB") 'crux-indent-rigidly-and-copy-to-clipboard)
  (global-set-key (kbd "C-c i") 'imenu-anywhere)
  )
(global-set-key (kbd "C-+") 'text-scale-increase)
(global-set-key (kbd "C--") 'text-scale-decrease)
(global-set-key (kbd "M-/") 'hippie-expand)
```


## proxy {#proxy}

```emacs-lisp
  (defun my/show-proxy ()
  "Show http/https proxy."
  (interactive)
  (if url-proxy-services
      (message "proxy is on" )
    (message "No proxy")))
(defun my/set-proxy ()
  "Set http/https proxy."
  (interactive)
  (setq url-proxy-services
      '(("no_proxy" . "^\\(localhost\\|10.*\\|192.168.*\\|172.*\\|127.0.0.1\\|::1\\)")
        ("http" . "127.0.0.1:61491")
        ("https" . "127.0.0.1:61491")))
  (my/show-proxy))
(defun my/unset-proxy ()
  "Unset http/https proxy."
  (interactive)
  (setq url-proxy-services nil)
  (my/show-proxy))
(defun my/toggle-proxy ()
  "Toggle http/https proxy."
  (interactive)
  (if url-proxy-services
      (unset-proxy)
    (my/set-proxy)))
```


## customized functions {#customized-functions}


### reselect {#reselect}

```emacs-lisp
(global-set-key (kbd "<f16>") 'set-markers-for-region)
(global-set-key (kbd "<S-f6>") 'set-region-from-markers)
(global-set-key (kbd "<s-S-f6>") 'unset-region-markers)
(defun set-markers-for-region ()
  (interactive)
  (make-local-variable 'm1)
  (make-local-variable 'm2)
  (setq m1 (copy-marker (mark)))
  (setq m2 (copy-marker (point)))
  (message "set-markers-for-region activated")
  )
(defun set-region-from-markers ()
  (interactive)
  (set-mark m1)
  (goto-char m2)
  (message "set-region-from-markers activated")
  )
(defun unset-region-markers ()
  (interactive)
  (set-marker m1 nil)
  (set-marker m2 nil)
  (message "unset-region-markers activated")
  )
```


### ifuns {#ifuns}


#### mannul install {#mannul-install}

```emacs-lisp
(load "~/.emacs.d/my-download/query-replace-many/query-replace-many.el")
```


#### window management {#window-management}

```emacs-lisp
(defun my/chatgpt ()
  (interactive)
  (progn
    (delete-other-windows)
    (split-window-right)
    (other-window 1)
    (chatgpt-shell)
      )
    )
```


#### others {#others}

```emacs-lisp
(defalias 'reload-buffer 'revert-buffer)
(defun say-region ()
  "Use macOS say command to speak the selected region."
  (interactive)
  (if (region-active-p)
      (shell-command-on-region (region-beginning) (region-end) "say")
    (message "No region selected")))
(defun my/copy-filename-and-line-number-to-kill-ring ()
  "Copy the current buffer's file name and the current line number to the kill ring."
  (interactive)
  (let* ((filename (buffer-file-name))
         (line-number (line-number-at-pos))
         (output (if filename
                     (format "%s:%d" filename line-number)
                   "Buffer does not have a file name.")))
    (kill-new output)
    (message "Copied to kill-ring: %s" output)))

(defun my/load-init-file ()
  "Load the init file"
  (interactive)
  (load-file "~/.emacs.d/init.el")
  )
(defun my/find-ssh-custom-file ()
  "Edit the `ssh/custom-file', in another window."
  (interactive)
  (let ((ssh-custom-file "~/.ssh/config"))
    (find-file-other-window ssh-custom-file)
    (message (format "opening %s" ssh-custom-file))
    )
  )
(defun my/find-zshell-custom-file ()
  "Edit the `ssh/custom-file', in another window."
  (interactive)
  (let ((zshell-custom-file "~/.zshrc"))
    (find-file-other-window zshell-custom-file)
    (message (format "opening %s" zshell-custom-file))
    )
  )
(defun my/find-python-custom-file ()
  "Edit the `ssh/custom-file', in another window."
  (interactive)
  (let ((zshell-custom-file "~/.ipython/profile_default/startup/ipython_init.py"))
    (find-file-other-window zshell-custom-file)
    (message (format "opening %s" zshell-custom-file))
    )
  )
(defun my/dired-tmp-dir ()
  "Edit the `ssh/custom-file', in another window."
  (interactive)
  (let ((tmp-dir "~/tmp/"))
    (dired tmp-dir)
    (message (format "opening %s" tmp-dir))
    )
  )
(defun my/set-cursor-yellow ()
  "Set cursor color to yellow"
  (interactive)
  (set-cursor-color "yellow")
  )
(defun my/org-remove-all-result-blocks ()
  (interactive)
  (save-excursion
    (goto-char (point-min))
    (while (search-forward "#+begin_src " nil t)
      (org-babel-remove-result))))
(global-set-key (kbd "s-i fs") 'my/find-ssh-custom-file)
(global-set-key (kbd "s-i fz") 'my/find-zshell-custom-file)
(global-set-key (kbd "s-i fp") 'my/find-python-custom-file)
(global-set-key (kbd "s-i ba") 'python-nav-beginning-of-block)
(global-set-key (kbd "s-i be") 'python-nav-end-of-block)
(global-set-key (kbd "s-i dt") 'my/dired-tmp-dir)
(global-set-key (kbd "s-i odr") 'my/org-remove-all-result-blocks)
(setq path-to-ctags "/opt/homebrew/opt/ctags/bin/ctags")
(defun create-tags (dir-name)
  "Create tags file."
  (interactive "DDirectory: ")
  (let ((icmd (format "/opt/homebrew/opt/ctags/bin/ctags -f %s/TAGS -e -R %s" (directory-file-name dir-name) (directory-file-name dir-name))))
    (message (format "I am creating tags for %s" dir-name))
    (message icmd)
    (shell-command icmd)
    ))
;; (format "%s -f TAGS -e -R %s" path-to-ctags (directory-file-name dir-name)))
(defun load-buffer ()
  "load current elisp buffer"
  (interactive)
  (load-file (buffer-file-name))
  )
(defun beautify-json ()
  (interactive)
  (let ((b (if mark-active (min (point) (mark)) (point-min)))
        (e (if mark-active (max (point) (mark)) (point-max))))
    (shell-command-on-region b e
                             "python -mjson.tool" (current-buffer) t)))
(defun eshell-other-window ()
  "Open a `shell' in a new window."
  (interactive)
  (let ((buf (eshell)))
    (switch-to-buffer (other-buffer buf))
    (switch-to-buffer-other-window buf)))
;; https://gitlab.mobvista.com/ad_algo/ltv_model/-/blob/main/setup.py
;; git@gitlab.mobvista.com:ad_algo/ltv_model.git
(defun clipboard/set (astring)
  "Copy a string to clipboard"
  (with-temp-buffer
    (insert astring)
    (clipboard-kill-region (point-min) (point-max))))
(defun my/git-share ()
  "share a git url for current buffer."
  (interactive)
  (progn
    (setq relative-file-name (file-relative-name buffer-file-name (projectile-project-root)))
    (setq icmd (format "python ~/bin/share_git.py %s" relative-file-name))
    (setq fin-path (substring
                    (shell-command-to-string icmd)
                    0 -1))
    (clipboard/set fin-path)
    (message (format "sharing-path is %s" fin-path))
    (browse-url fin-path)
    )
  )
(defun my/git-merge ()
  "share a git url for current buffer."
  (interactive)
  (progn
    (setq icmd (format "python ~/bin/git_merge_url.py"))
    (setq myurl (substring
                 (shell-command-to-string icmd)
                 0 -1))
    (clipboard/set myurl)
    (message myurl)
    (browse-url merge-url)
    )
  )
(defun my/git-merge2 ()
  "share a git url for current buffer."
  (interactive)
  (let*
      ((icmd (format "python ~/bin/git_merge_url.py"))
       (merge-url (substring
                   (shell-command-to-string icmd)
                   0 -1)))
    (clipboard/set merge-url)
    ;; (message (format "merge url is %s" merge-url))
    (browse-url merge-url)
    )
  )
(defun my/select-current-line ()
  "Select the current line under the cursor."
  (interactive)
  (end-of-line)               ; Move to the end of the line
  (set-mark (line-beginning-position)) ; Set the mark at the beginning
  (activate-mark); Activate the selection
        ;;; ifuns.el ends here
  )
(global-set-key (kbd "s-i sl") 'my/select-current-line)
```


## trading {#trading}

```emacs-lisp
(use-package pine-script-mode
  :ensure t
  :pin melpa-stable
  :mode (("\\.pine" . pine-script-mode)))
```


## final {#final}

```emacs-lisp
(defvar my-keys-minor-mode-map
  (let ((map (make-sparse-keymap)))
    ;; (define-key projectile-rails-mode-map (kbd "C-c r") 'projectile-rails-command-map)
    (define-key map  (kbd "C-x g") 'magit-status)
    (define-key map  (kbd "C-s") 'isearch-forward)
    (define-key map (kbd "C-c SPC") 'ace-jump-mode)
    (define-key map (kbd "C-x SPC") 'rectangle-mark-mode)
    (global-set-key (kbd "C-c g") 'google-this-mode-submap)
    ;; (move-text-default-bindings)
    map)
  "my-keys-minor-mode keymap.")
(define-minor-mode my-mode
  "A minor mode so that my key settings override annoying major modes."
  :init-value t
  :lighter  my-mode
  :keymap my-keys-minor-mode-map
  )
(my-mode 1)
(setq inhibit-startup-screen t)
(setq initial-buffer-choice t)
(add-to-list 'default-frame-alist '(fullscreen . maximized))
(add-hook 'emacs-startup-hook 'delete-other-windows)
```

