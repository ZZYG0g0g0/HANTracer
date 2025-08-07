import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import spoon.Launcher;
import spoon.reflect.CtModel;
import spoon.reflect.code.CtInvocation;
import spoon.reflect.declaration.CtClass;
import spoon.reflect.declaration.CtImport;
import spoon.reflect.declaration.CtType;
import spoon.reflect.reference.CtTypeReference;
import spoon.reflect.visitor.filter.TypeFilter;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

public class SpoonExample {

    public static void main(String[] args) {
        Launcher launcher = new Launcher();
        List<File> javaFiles = listJavaFiles(new File("C:\\Users\\zzy\\Desktop\\benchmark file\\republic\\CodeDependencyTest\\src\\main\\new cc\\smos"));

        for (File file : javaFiles) {
            launcher.addInputResource(file.getAbsolutePath());
        }

        launcher.buildModel();
        CtModel model = launcher.getModel();

        // 获取所有类名集合，用于过滤调用关系
        Set<String> classNames = model.getElements(new TypeFilter<>(CtClass.class)).stream()
                .map(CtType::getQualifiedName)
                .collect(Collectors.toSet());

        List<String[]> relationships = new ArrayList<>();


        // 提取继承关系
        for (CtClass<?> ctClass : model.getElements(new TypeFilter<>(CtClass.class))) {
            CtTypeReference<?> superClassRef = ctClass.getSuperclass();
            if (superClassRef != null) {
                String childClass = ctClass.getQualifiedName();
                String parentClass = superClassRef.getQualifiedName();
                if (classNames.contains(parentClass) && !isInnerClass(childClass) && !isInnerClass(parentClass)) {
                    relationships.add(new String[]{stripPackageName(childClass), stripPackageName(parentClass), "extend"});
                }
            }
        }

// 提取实现关系（implements）
        for (CtClass<?> ctClass : model.getElements(new TypeFilter<>(CtClass.class))) {
            for (CtTypeReference<?> iface : ctClass.getSuperInterfaces()) {
                String childClass = ctClass.getQualifiedName();
                String interfaceClass = iface.getQualifiedName();
                if (classNames.contains(interfaceClass) && !isInnerClass(childClass) && !isInnerClass(interfaceClass)) {
                    relationships.add(new String[]{stripPackageName(childClass), stripPackageName(interfaceClass), "implement"});
                }
            }
        }

// 用于去重的方法调用记录集合
        Set<String> methodCallSet = new HashSet<>();

// 提取方法调用关系（Method Call），剔除同类方法调用并去重
        for (CtClass<?> ctClass : model.getElements(new TypeFilter<>(CtClass.class))) {
            String caller = ctClass.getQualifiedName();

            ctClass.getElements(new TypeFilter<>(CtInvocation.class)).forEach(invocation -> {
                CtTypeReference<?> targetType = invocation.getTarget() != null ? invocation.getTarget().getType() : null;

                if (targetType != null && classNames.contains(targetType.getQualifiedName())) {
                    String callee = targetType.getQualifiedName();

                    // 忽略同类调用
                    if (!caller.equals(callee) && !isInnerClass(caller) && !isInnerClass(callee)) {
                        String key = caller + "->" + callee;

                        if (!methodCallSet.contains(key)) {
                            methodCallSet.add(key);
                            relationships.add(new String[]{stripPackageName(caller), stripPackageName(callee), "call"});
                        }
                    }
                }
            });
        }

        // 提取类调用关系（import语句）
        for (CtClass<?> ctClass : model.getElements(new TypeFilter<>(CtClass.class))) {
            Set<CtImport> imports = new HashSet<>(ctClass.getPosition().getCompilationUnit().getImports());
            for (CtImport imp : imports) {
                if (imp.getReference() instanceof CtTypeReference) {
                    String importedClassName = ((CtTypeReference<?>) imp.getReference()).getQualifiedName();
                    if (classNames.contains(importedClassName)) {
                        String className1 = ctClass.getQualifiedName();
                        String className2 = importedClassName;
                        if (!isInnerClass(className1) && !isInnerClass(className2)) {
                            relationships.add(new String[]{stripPackageName(className1), stripPackageName(className2), "import"});
                        }
                    }
                }
            }
        }

        // 创建Excel文件
        createExcelFile(relationships);
    }

    private static List<File> listJavaFiles(File folder) {
        List<File> javaFiles = new ArrayList<>();
        for (File file : folder.listFiles()) {
            if (file.isDirectory()) {
                javaFiles.addAll(listJavaFiles(file));
            } else if (file.getName().endsWith(".java")) {
                javaFiles.add(file);
            }
        }
        return javaFiles;
    }

    private static String stripPackageName(String qualifiedName) {
        int lastDotIndex = qualifiedName.lastIndexOf('.');
        if (lastDotIndex == -1) {
            return qualifiedName;
        }
        return qualifiedName.substring(lastDotIndex + 1);
    }

    private static boolean isInnerClass(String className) {
        return className.contains("$");
    }

    private static void createExcelFile(List<String[]> relationships) {
        Workbook workbook = new XSSFWorkbook();
        Sheet sheet = workbook.createSheet("Class Relationships");

        // 创建标题行
        Row headerRow = sheet.createRow(0);
        headerRow.createCell(0).setCellValue("Class 1");
        headerRow.createCell(1).setCellValue("Class 2");
        headerRow.createCell(2).setCellValue("Relationship");

        // 填充数据行
        int rowNum = 1;
        for (String[] relationship : relationships) {
            Row row = sheet.createRow(rowNum++);
            row.createCell(0).setCellValue(relationship[0]);
            row.createCell(1).setCellValue(relationship[1]);
            row.createCell(2).setCellValue(relationship[2]);
        }

        // 写入Excel文件
        try (FileOutputStream fileOut = new FileOutputStream("C:\\Users\\zzy\\Desktop\\benchmark file\\republic\\CodeDependencyTest\\ESE_CodeDependency\\smos.xlsx")) {
            workbook.write(fileOut);
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                workbook.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}