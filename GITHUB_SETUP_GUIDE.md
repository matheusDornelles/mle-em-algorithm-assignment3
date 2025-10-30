# 🚀 Guia para Upload do Repositório para GitHub

## 📋 Pré-requisitos
- ✅ Conta no GitHub (se não tiver: https://github.com/join)
- ✅ Git instalado localmente (verificar com: `git --version`)

## 🔄 Passo a Passo para GitHub

### 1. **Criar Repositório no GitHub**
1. Acesse: https://github.com/new
2. **Nome do repositório**: `mle-em-algorithm-assignment3` (ou outro nome de sua escolha)
3. **Descrição**: `Maximum Likelihood Estimation & EM Algorithm implementation with missing data analysis`
4. **Visibilidade**: 
   - 🔓 **Public** (recomendado para portfólio)
   - 🔒 **Private** (se preferir manter privado)
5. ❌ **NÃO** marque "Add a README file" (já temos um)
6. ❌ **NÃO** adicione .gitignore (já temos um)
7. ❌ **NÃO** adicione license (já temos uma)
8. Clique **"Create repository"**

### 2. **Conectar Repositório Local com GitHub**

Após criar no GitHub, você verá uma página com comandos. Use estes comandos no seu terminal:

```powershell
# Adicionar o repositório remoto (substitua SEU-USUARIO pelo seu username)
git remote add origin https://github.com/SEU-USUARIO/mle-em-algorithm-assignment3.git

# Renomear branch para 'main' (padrão atual do GitHub)
git branch -M main

# Fazer upload dos arquivos
git push -u origin main
```

### 3. **Comandos Prontos para Copiar/Colar**

**Execute estes comandos no PowerShell (na pasta assignment3):**

```powershell
# 1. Adicionar repositório remoto (SUBSTITUA SEU-USUARIO)
git remote add origin https://github.com/SEU-USUARIO/mle-em-algorithm-assignment3.git

# 2. Configurar branch principal
git branch -M main

# 3. Upload inicial
git push -u origin main
```

### 4. **Se Houver Problemas de Autenticação**

O GitHub não aceita mais senha. Use uma dessas opções:

#### Opção A: Personal Access Token
1. Vá em: GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Clique "Generate new token (classic)"
3. Selecione escopo "repo"
4. Copie o token gerado
5. Use o token como senha quando solicitado

#### Opção B: GitHub CLI (Recomendado)
```powershell
# Instalar GitHub CLI
winget install --id GitHub.cli

# Fazer login
gh auth login

# Criar e conectar repositório automaticamente
gh repo create mle-em-algorithm-assignment3 --public --source=. --remote=origin --push
```

## 🔄 Workflow para Futuras Atualizações

Quando você fizer mudanças no código:

```powershell
# 1. Ver arquivos modificados
git status

# 2. Adicionar mudanças
git add .

# 3. Commit com mensagem descritiva
git commit -m "Add new feature: advanced visualization analysis"

# 4. Upload para GitHub
git push
```

## 🌟 Alternativas ao GitHub

### GitLab
1. Criar conta em: https://gitlab.com
2. Novo projeto: https://gitlab.com/projects/new
3. Comandos similares (trocar github.com por gitlab.com)

### Bitbucket  
1. Criar conta em: https://bitbucket.org
2. Criar repositório
3. Comandos similares (trocar github.com por bitbucket.org)

## 🎯 Próximos Passos Recomendados

Após o upload:

### 1. **Adicionar Badges ao README**
O README já inclui badges, mas você pode personalizar:
- Status do build
- Número de estrelas  
- Última atualização
- Linguagem principal

### 2. **Configurar GitHub Pages** (Opcional)
Para hospedar uma página web com seus resultados:
```powershell
# Criar branch gh-pages
git checkout --orphan gh-pages
git rm -rf .
echo "<h1>MLE & EM Algorithm Results</h1>" > index.html
git add index.html
git commit -m "Initial GitHub Pages"
git push origin gh-pages
```

### 3. **Adicionar Tópicos/Tags**
No GitHub, vá em Settings → General → Topics:
- `machine-learning`
- `expectation-maximization`
- `maximum-likelihood`
- `missing-data`
- `python`
- `statistical-analysis`

## 🛟 Resolução de Problemas

### Erro: "Repository already exists"
```powershell
git remote -v  # Ver repositórios remotos
git remote remove origin  # Remover se existir
# Adicionar novamente com URL correta
```

### Erro: "Permission denied"
- Verificar se o username/repositório estão corretos
- Usar Personal Access Token em vez de senha
- Verificar se tem permissão de escrita no repositório

### Arquivo muito grande
O GitHub tem limite de 100MB por arquivo. Se necessário:
```powershell
# Ver arquivos grandes
find . -size +50M

# Usar Git LFS para arquivos grandes
git lfs track "*.png"
git add .gitattributes
```

## 📞 Suporte

- **GitHub Docs**: https://docs.github.com
- **Git Documentation**: https://git-scm.com/doc
- **GitHub Community**: https://github.community

---

✅ **Checklist Final**
- [ ] Repositório criado no GitHub
- [ ] Arquivos uploaded com sucesso
- [ ] README.md sendo exibido corretamente
- [ ] Imagens carregando nas visualizações
- [ ] Código executável (requirements.txt funcionando)

🎉 **Parabéns! Seu projeto agora está online e pode ser compartilhado com o mundo!**